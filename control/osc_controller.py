"""
Operational Space Controller (OSC) for Franka Panda .

Works WITH scene.xml general actuators (affine bias PD):
    force = gain * (ctrl - qpos) - damping_bias * qvel

Architecture:
    1. Task-space PD: F = Kp * (target - ee) - Kd * ee_vel
    2. Joint-space via damped pseudoinverse: dq = J_pinv @ F
    3. q_target = q_current + dq → send to actuators
    4. Null-space posture bias keeps arm well-configured

"""

import numpy as np
import mujoco


class OSCController:
    HOME_QPOS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

    def __init__(
        self,
        model:          mujoco.MjModel,
        data:           mujoco.MjData,
        ee_site_name:   str   = "ee_center_site",
        joint_names:    list  = None,
        kp:             float = 150.0,
        kd:             float = 25.0,
        damping:        float = 1e-4,
        max_dq:         float = 0.15,
        null_stiffness: float = 5.0,
    ):
        self.model = model
        self.data  = data
        self.kp    = kp
        self.kd    = kd
        self.damping        = damping
        self.max_dq         = max_dq
        self.null_stiffness = null_stiffness

        self._ee_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
        assert self._ee_site_id >= 0, f"Site '{ee_site_name}' not found"

        if joint_names is None:
            joint_names = [f"joint{i}" for i in range(1, 8)]

        self._joint_ids      = []
        self._joint_qpos_ids = []
        self._joint_dof_ids  = []
        for name in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert jid >= 0, f"Joint '{name}' not found"
            self._joint_ids.append(jid)
            self._joint_qpos_ids.append(model.jnt_qposadr[jid])
            self._joint_dof_ids.append(model.jnt_dofadr[jid])

        self.n_joints = len(joint_names)
        self.nv = model.nv
        self._jacp = np.zeros((3, self.nv))

    def compute(self, target_pos: np.ndarray) -> np.ndarray:
        """
        Compute joint position targets to move EE toward target_pos.

        Returns:
            q_target: (7,) joint positions for actuators 0-6
        """
        ee_pos = self.data.site_xpos[self._ee_site_id]

        self._jacp[:] = 0
        mujoco.mj_jacSite(self.model, self.data, self._jacp, None,
                          self._ee_site_id)
        ee_vel = self._jacp @ self.data.qvel

        # Task-space PD
        F = self.kp * (target_pos - ee_pos) - self.kd * ee_vel

        # Arm-only Jacobian (3x7)
        dof_ids = [self._joint_dof_ids[i] for i in range(self.n_joints)]
        J = self._jacp[:, dof_ids]

        # Damped pseudoinverse
        JJT = J @ J.T + self.damping * np.eye(3)
        J_pinv = J.T @ np.linalg.solve(JJT, np.eye(3))

        dq = J_pinv @ F

        # Null-space posture bias
        if self.null_stiffness > 0:
            q_current = self.get_joint_positions()
            q_error = self.HOME_QPOS - q_current
            N = np.eye(self.n_joints) - J_pinv @ J
            dq += self.null_stiffness * (N @ q_error)

        # Per-joint clamp
        dq = np.clip(dq, -self.max_dq, self.max_dq)

        # Target = current + delta, clamped to limits
        q_current = self.get_joint_positions()
        q_target = q_current + dq
        for i in range(self.n_joints):
            lo = self.model.jnt_range[self._joint_ids[i], 0]
            hi = self.model.jnt_range[self._joint_ids[i], 1]
            q_target[i] = np.clip(q_target[i], lo, hi)

        return q_target

    #  Getters

    def get_ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._ee_site_id].copy()

    def get_ee_vel(self) -> np.ndarray:
        self._jacp[:] = 0
        mujoco.mj_jacSite(self.model, self.data, self._jacp, None,
                          self._ee_site_id)
        return self._jacp @ self.data.qvel

    def get_joint_positions(self) -> np.ndarray:
        return np.array([
            self.data.qpos[self._joint_qpos_ids[i]]
            for i in range(self.n_joints)
        ])

    def get_joint_velocities(self) -> np.ndarray:
        return np.array([
            self.data.qvel[self._joint_dof_ids[i]]
            for i in range(self.n_joints)
        ])


#  Smoke test 

if __name__ == "__main__":
    import os
    import time

    scene_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "assets", "scene.xml"
    )

    print("Loading model...")
    model = mujoco.MjModel.from_xml_path(scene_path)
    data  = mujoco.MjData(model)

    #  CRITICAL: Disable weld at RUNTIME 
    for i in range(model.neq):
        if model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
            model.eq_active0[i] = 0   # default
            data.eq_active[i] = 0     # runtime — THIS is what matters
            print(f"  Disabled weld constraint {i} (both model + data)")

    #  Set home pose 
    home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    joint_names = [f"joint{i}" for i in range(1, 8)]

    def init_data(d):
        """Set home pose and ctrl. No warmup needed — actuators hold fine."""
        for i, name in enumerate(joint_names):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            d.qpos[model.jnt_qposadr[jid]] = home[i]
            d.qvel[model.jnt_dofadr[jid]] = 0.0
        for i in range(7):
            d.ctrl[i] = home[i]
        d.ctrl[7] = 0.04
        d.ctrl[8] = 0.04
        # Disable weld on this data instance too
        for i in range(model.neq):
            if model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                d.eq_active[i] = 0
        mujoco.mj_forward(model, d)

    init_data(data)

    controller = OSCController(model, data, ee_site_name="ee_center_site",
                               kp=150.0, kd=25.0)
    ee_start = controller.get_ee_pos()
    print(f"EE start: {ee_start}")

    results = {}

    #  Test 1: Move 10cm right 
    print("\n=== Test 1: Move 10cm right ===")
    target1 = ee_start.copy()
    target1[1] += 0.10
    print(f"  Target: {target1}")

    for step in range(500):
        jt = controller.compute(target1)
        for i in range(7):
            data.ctrl[i] = jt[i]
        mujoco.mj_step(model, data)
        if step % 100 == 0:
            ee = controller.get_ee_pos()
            d = np.linalg.norm(ee - target1)
            print(f"  Step {step:3d}: dist={d:.4f}")

    d1 = np.linalg.norm(controller.get_ee_pos() - target1)
    results["10cm_right"] = d1 < 0.02
    print(f"  Final: {d1:.4f} {'PASS ' if d1 < 0.02 else 'FAIL '}")

    #  Test 2: Ball area 
    print("\n=== Test 2: Ball area [0.45, 0.1, 0.48] ===")
    data2 = mujoco.MjData(model)
    init_data(data2)
    ctrl2 = OSCController(model, data2, ee_site_name="ee_center_site",
                          kp=150.0, kd=25.0)

    target2 = np.array([0.45, 0.10, 0.48])
    for step in range(500):
        jt = ctrl2.compute(target2)
        for i in range(7):
            data2.ctrl[i] = jt[i]
        mujoco.mj_step(model, data2)
        if step % 100 == 0:
            d = np.linalg.norm(ctrl2.get_ee_pos() - target2)
            print(f"  Step {step:3d}: dist={d:.4f}")

    d2 = np.linalg.norm(ctrl2.get_ee_pos() - target2)
    results["ball_area"] = d2 < 0.02
    print(f"  Final: {d2:.4f} {'PASS ' if d2 < 0.02 else 'FAIL '}")

    #  Test 3: Sequential waypoints (pick-and-place path)
    print("\n=== Test 3: Sequential waypoints ===")
    data3 = mujoco.MjData(model)
    init_data(data3)
    ctrl3 = OSCController(model, data3, ee_site_name="ee_center_site",
                          kp=150.0, kd=25.0)

    waypoints = [
        ("above ball",  np.array([0.45, 0.0, 0.55]),  400),
        ("at ball",     np.array([0.45, 0.0, 0.45]),  400),
        ("lift",        np.array([0.45, 0.0, 0.55]),  400),
        ("above bin",   np.array([0.72, 0.18, 0.55]), 500),
        ("at bin",      np.array([0.72, 0.18, 0.45]), 400),
    ]

    wp_ok = True
    for wp_name, wp_target, wp_steps in waypoints:
        for step in range(wp_steps):
            jt = ctrl3.compute(wp_target)
            for i in range(7):
                data3.ctrl[i] = jt[i]
            mujoco.mj_step(model, data3)

        ee = ctrl3.get_ee_pos()
        d = np.linalg.norm(ee - wp_target)
        ok = d < 0.03
        if not ok:
            wp_ok = False
        print(f"  {wp_name:12s}: dist={d:.4f} {'' if ok else ''}")

    results["waypoints"] = wp_ok

    #  Test 4: Timing 
    print("\n=== Test 4: Timing ===")
    times = []
    for _ in range(1000):
        t0 = time.time()
        _ = ctrl3.compute(np.array([0.5, 0.0, 0.5]))
        times.append((time.time() - t0) * 1e6)
    print(f"  avg={np.mean(times):.0f}μs, max={np.max(times):.0f}μs")

    #  Test 5: Gripper 
    print("\n=== Test 5: Gripper ===")
    f1_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
    f2_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")
    f1_adr = model.jnt_qposadr[f1_jid]
    f2_adr = model.jnt_qposadr[f2_jid]

    data3.ctrl[7] = 0.0
    data3.ctrl[8] = 0.0
    for _ in range(200):
        mujoco.mj_step(model, data3)
    f1 = data3.qpos[f1_adr]
    f2 = data3.qpos[f2_adr]
    print(f"  Closed: f1={f1:.4f}, f2={f2:.4f}")

    data3.ctrl[7] = 0.04
    data3.ctrl[8] = 0.04
    for _ in range(200):
        mujoco.mj_step(model, data3)
    f1 = data3.qpos[f1_adr]
    f2 = data3.qpos[f2_adr]
    print(f"  Opened: f1={f1:.4f}, f2={f2:.4f}")
    results["gripper"] = f1 > 0.035 and f2 > 0.035
    print(f"  {'PASS ' if results['gripper'] else 'FAIL '}")

    #  Test 6: Rapid direction changes (RL stress test) 
    print("\n=== Test 6: Rapid direction changes (simulates RL exploration) ===")
    data6 = mujoco.MjData(model)
    init_data(data6)
    ctrl6 = OSCController(model, data6, ee_site_name="ee_center_site",
                          kp=150.0, kd=25.0)

    rng = np.random.default_rng(42)
    ee_positions = []
    for step in range(1000):
        # Random target in workspace every 50 steps
        if step % 50 == 0:
            target = np.array([
                rng.uniform(0.30, 0.75),
                rng.uniform(-0.25, 0.25),
                rng.uniform(0.43, 0.60),
            ])
        jt = ctrl6.compute(target)
        for i in range(7):
            data6.ctrl[i] = jt[i]
        mujoco.mj_step(model, data6)
        ee_positions.append(ctrl6.get_ee_pos().copy())

    ee_positions = np.array(ee_positions)
    # Check: no NaN, no explosion, stays in workspace
    no_nan = not np.any(np.isnan(ee_positions))
    in_workspace = (np.all(ee_positions[:, 0] > -0.5) and
                    np.all(ee_positions[:, 0] < 1.5) and
                    np.all(ee_positions[:, 2] > 0.0) and
                    np.all(ee_positions[:, 2] < 1.5))
    results["rl_stress"] = no_nan and in_workspace
    print(f"  No NaN: {no_nan}")
    print(f"  In workspace: {in_workspace}")
    print(f"  EE range: x=[{ee_positions[:,0].min():.3f},{ee_positions[:,0].max():.3f}] "
          f"y=[{ee_positions[:,1].min():.3f},{ee_positions[:,1].max():.3f}] "
          f"z=[{ee_positions[:,2].min():.3f},{ee_positions[:,2].max():.3f}]")
    print(f"  {'PASS ' if results['rl_stress'] else 'FAIL '}")

    #  Summary 
    print(f"\n{'='*60}")
    passed = sum(results.values())
    total = len(results)
    for name, ok in results.items():
        print(f"  {name:20s}: {'PASS ' if ok else 'FAIL '}")
    print(f"\n  {passed}/{total} passed")
    if passed == total:
        print("  ALL TESTS PASSED ")
    else:
        print("  SOME TESTS FAILED ")