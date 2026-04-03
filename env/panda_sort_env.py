"""
PandaSortEnv — multi-ball capable.

Observation redesign: encodes ALL balls, not just one target.
This prevents obs discontinuity when a ball is sorted.

Observation (per-ball encoding, fixed N_MAX_BALLS=4 slots):
    ee_pos          (3)   EE xyz
    ee_vel          (3)   EE Cartesian velocity
    finger_width    (1)   sum of both finger positions
    --- per ball (*4 slots) ---
    ball_pos        (3)   ball xyz
    ball_rel        (3)   ball pos relative to EE
    ball_color_bin  (3)   correct bin xyz for this ball
    ball_sorted     (1)   1.0 if this ball is sorted
    ball_active     (1)   1.0 if this slot is active (has a ball)
    --- end per ball (11 * 4 = 44) ---
    total: 7 + 44 = 51D

Action (4D):
    [dx, dy, dz, gripper] in [-1, 1]

The policy sees all balls simultaneously and learns to sequence them.
No observation discontinuity when a ball is sorted.
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from control.osc_controller import OSCController

# Constants

SCENE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets", "scene.xml"
)

HOME_QPOS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

BIN_POSITIONS = {
    "red":  np.array([0.72, -0.18, 0.421]),
    "blue": np.array([0.72,  0.18, 0.421]),
}
BIN_RADIUS    = 0.07
TABLE_Z       = 0.42
BALL_RADIUS   = 0.025
LIFT_THRESH   = 0.08
POS_SCALE     = 0.05
N_SUBSTEPS    = 50
N_MAX_BALLS   = 4       # max ball slots in obs (matches XML)
PER_BALL_DIM  = 11      # pos(3) + rel(3) + bin(3) + sorted(1) + active(1)
OBS_DIM       = 7 + N_MAX_BALLS * PER_BALL_DIM  # 7 + 44 = 51

BALL_COLORS = {
    "red":  np.array([0.85, 0.15, 0.15, 1.0]),
    "blue": np.array([0.15, 0.15, 0.85, 1.0]),
}

WS_LOW  = np.array([0.15, -0.35, 0.42])
WS_HIGH = np.array([0.85,  0.35, 0.65])


class PandaSortEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        n_balls:           int  = 1,
        reward_mode:       str  = "dense",
        max_episode_steps: int  = 200,
        domain_randomize:  bool = False,
        render_mode:       str  = None,
        seed:              Optional[int] = None,
    ):
        super().__init__()
        assert n_balls <= N_MAX_BALLS, f"n_balls={n_balls} > N_MAX_BALLS={N_MAX_BALLS}"
        self.n_balls           = n_balls
        self.reward_mode       = reward_mode
        self.max_episode_steps = max_episode_steps
        self.domain_randomize  = domain_randomize
        self.render_mode       = render_mode
        self.rng               = np.random.default_rng(seed)

        # MuJoCo
        self.model = mujoco.MjModel.from_xml_path(SCENE_PATH)
        self.data  = mujoco.MjData(self.model)

        self._cache_ids()

        self.osc = OSCController(
            self.model, self.data,
            ee_site_name="ee_center_site",
            kp=150.0, kd=25.0,
            max_dq=0.05, null_stiffness=5.0,
        )

        # Spaces 
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(4,), dtype=np.float32)

        # Episode state
        self.ball_colors_ep = []
        self._step_count = 0
        self._prev_n_sorted = 0
        self._ball_max_height = [0.0] * N_MAX_BALLS

    # Reset

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._step_count = 0
        self._prev_n_sorted = 0
        self._ball_max_height = [0.0] * N_MAX_BALLS

        mujoco.mj_resetData(self.model, self.data)


        for i in range(7):
            self.data.qpos[self._joint_qpos[i]] = HOME_QPOS[i]
            self.data.qvel[self._joint_dof[i]] = 0.0
            self.data.ctrl[i] = HOME_QPOS[i]
        self.data.ctrl[7] = 0.04
        self.data.ctrl[8] = 0.04
        self.data.qpos[self._f1_adr] = 0.04
        self.data.qpos[self._f2_adr] = 0.04

        # Randomize initial gripper state so agent learns to open before grasping
        random_fw = self.rng.uniform(0.0, 0.04)
        self.data.ctrl[7] = random_fw
        self.data.ctrl[8] = random_fw
        self.data.qpos[self._f1_adr] = random_fw
        self.data.qpos[self._f2_adr] = random_fw

        self.ball_colors_ep = [
            self.rng.choice(["red", "blue"]) for _ in range(self.n_balls)
        ]
        self._place_balls()
        self._set_ball_colors()

        for i in range(self.n_balls, N_MAX_BALLS):
            self._hide_ball(i)

        if self.domain_randomize:
            self._apply_domain_rand()

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = {
            "ball_colors": self.ball_colors_ep,
            "reward_mode": self.reward_mode,
        }
        return obs, info

    # Step 

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        self._apply_action(action)
        self._step_count += 1

        for i in range(self.n_balls):
            bp = self._get_ball_pos(i)
            self._ball_max_height[i] = max(self._ball_max_height[i], bp[2])

        obs = self._get_obs()
        reward = self._compute_reward(obs)
        n_sorted = self._count_sorted()
        n_in_bin = sum(self._ball_in_bin(i) for i in range(self.n_balls))
        success = n_sorted >= self.n_balls
        trunc = self._step_count >= self.max_episode_steps

        info = {
            "success":        success,
            "n_sorted":       n_sorted,
            "n_in_bin":       n_in_bin,
            "n_sorted_legit": n_sorted,
            "max_heights":    [self._ball_max_height[i] for i in range(self.n_balls)],
            "n_lifted":       sum(1 for i in range(self.n_balls)
                                 if self._ball_max_height[i] > TABLE_Z + BALL_RADIUS + LIFT_THRESH),
            "reward_mode":    self.reward_mode,
            "step":           self._step_count,
        }

        self._prev_n_sorted = n_sorted
        return obs, reward, success, trunc, info

    # Action 

    def _apply_action(self, action: np.ndarray):
        dx, dy, dz = action[0] * POS_SCALE, action[1] * POS_SCALE, action[2] * POS_SCALE
        grip_delta = action[3]

        current_fw = self.data.qpos[self._f1_adr]
        new_fw = np.clip(current_fw + grip_delta * 0.01, 0.0, 0.04)
        self.data.ctrl[7] = new_fw
        self.data.ctrl[8] = new_fw

        ee_pos = self.osc.get_ee_pos()
        target = ee_pos + np.array([dx, dy, dz])
        target = np.clip(target, WS_LOW, WS_HIGH)

        for _ in range(N_SUBSTEPS):
            jt = self.osc.compute(target)
            for i in range(7):
                self.data.ctrl[i] = jt[i]
            mujoco.mj_step(self.model, self.data)

    # Observation

    def _get_obs(self) -> np.ndarray:
        """
        51D observation:
            ee_pos(3) + ee_vel(3) + finger_width(1) = 7
            + 4 ball slots × 11D each = 44
            total = 51

        Each ball slot:
            ball_pos(3) + ball_rel(3) + bin_pos(3) + sorted(1) + active(1)

        Inactive slots (ball index >= n_balls) are all zeros.
        Sorted balls remain visible with sorted=1.0 so the policy
        knows they're done and doesn't re-target them.
        """
        ee_pos = self.osc.get_ee_pos()
        ee_vel = self.osc.get_ee_vel()
        fw = self.data.qpos[self._f1_adr] + self.data.qpos[self._f2_adr]

        # Robot state: 7D
        robot_obs = np.concatenate([ee_pos, ee_vel, [fw]])

        # Per-ball encoding: N_MAX_BALLS × 11D
        ball_obs = np.zeros(N_MAX_BALLS * PER_BALL_DIM, dtype=np.float32)

        for i in range(N_MAX_BALLS):
            offset = i * PER_BALL_DIM

            if i < self.n_balls:
                bp = self._get_ball_pos(i)
                color = self.ball_colors_ep[i]
                bin_pos = BIN_POSITIONS[color]
                is_sorted = float(self._ball_is_sorted(i))

                ball_obs[offset + 0:offset + 3]  = bp                # ball_pos
                ball_obs[offset + 3:offset + 6]  = bp - ee_pos       # ball_rel
                ball_obs[offset + 6:offset + 9]  = bin_pos           # target bin
                ball_obs[offset + 9]             = is_sorted         # sorted flag
                ball_obs[offset + 10]            = 1.0               # active flag
            # else: all zeros (inactive slot)

        obs = np.concatenate([robot_obs, ball_obs]).astype(np.float32)
        return obs

    # Reward 

    def _compute_reward(self, obs: np.ndarray) -> float:
        if self.reward_mode == "sparse":
            return self._sparse_reward()
        return self._dense_reward(obs)

    def _dense_reward(self, obs: np.ndarray) -> float:
        """
        Dense reward that works for any number of balls.

        For each UNSORTED ball, compute shaping toward nearest unsorted ball.
        Only the closest unsorted ball gets shaping (prevents split attention).
        Placement bonus for each ball sorted. Success bonus when all done.
        """
        ee_pos = obs[0:3]
        fw     = obs[6]

        r = 0.0

        # Find closest unsorted ball for shaping
        best_idx = None
        best_dist = np.inf
        for i in range(self.n_balls):
            offset = 7 + i * PER_BALL_DIM
            is_sorted = obs[offset + 9]
            if is_sorted > 0.5:
                continue
            ball_rel = obs[offset + 3:offset + 6]
            d = np.linalg.norm(ball_rel)
            if d < best_dist:
                best_dist = d
                best_idx = i

        if best_idx is not None:
            offset = 7 + best_idx * PER_BALL_DIM
            ball_pos = obs[offset + 0:offset + 3]
            ball_rel = obs[offset + 3:offset + 6]
            bin_pos  = obs[offset + 6:offset + 9]

            dist_ee_ball  = np.linalg.norm(ball_rel)
            dist_ball_bin = np.linalg.norm(ball_pos[:2] - bin_pos[:2])
            lift_amount   = ball_pos[2] - (TABLE_Z + BALL_RADIUS)

            # Ball in gripper check
            ball_in_grip = float(dist_ee_ball < 0.04 and fw < 0.06 and fw > 0.01)

            # Stage 1: Reach
            r += 0.1 * (1.0 - np.tanh(5.0 * dist_ee_ball))

            # Stage 2: Grasp
            if dist_ee_ball < 0.05:
                r += 0.1 * ball_in_grip

            # Stage 3: Lift
            if ball_in_grip > 0.5:
                r_lift = float(np.clip(lift_amount / LIFT_THRESH, 0.0, 1.0))
                r += 0.05 * r_lift

                # Stage 4: Transport
                if lift_amount > LIFT_THRESH * 0.5:
                    r += 0.08 * (1.0 - np.tanh(5.0 * dist_ball_bin))

        # Sparse bonuses 
        n_sorted_now = self._count_sorted()
        new_sorted = n_sorted_now - self._prev_n_sorted
        if new_sorted > 0:
            r += 50.0 * new_sorted

        if n_sorted_now >= self.n_balls:
            steps_remaining = self.max_episode_steps - self._step_count
            r += 100.0 + steps_remaining * 0.5

        return float(r)

    def _sparse_reward(self) -> float:
        n_sorted = self._count_sorted()
        new_sorted = n_sorted - self._prev_n_sorted
        r = float(new_sorted) * 1.0
        if n_sorted >= self.n_balls:
            r += 5.0
        return r

    # Ball management 

    def _ball_is_sorted(self, idx: int) -> bool:
        bp = self._get_ball_pos(idx)
        color = self.ball_colors_ep[idx]
        bin_pos = BIN_POSITIONS[color]
        dist_xy = np.linalg.norm(bp[:2] - bin_pos[:2])
        on_surface = bp[2] < TABLE_Z + 0.06
        in_bin = dist_xy < BIN_RADIUS and on_surface
        was_lifted = self._ball_max_height[idx] > TABLE_Z + BALL_RADIUS + LIFT_THRESH
        return in_bin and was_lifted

    def _ball_in_bin(self, idx: int) -> bool:
        bp = self._get_ball_pos(idx)
        color = self.ball_colors_ep[idx]
        bin_pos = BIN_POSITIONS[color]
        dist_xy = np.linalg.norm(bp[:2] - bin_pos[:2])
        on_surface = bp[2] < TABLE_Z + 0.06
        return dist_xy < BIN_RADIUS and on_surface

    def _count_sorted(self) -> int:
        return sum(self._ball_is_sorted(i) for i in range(self.n_balls))

    def _place_balls(self):
        placed = []
        for i in range(self.n_balls):
            for _ in range(100):
                x = self.rng.uniform(0.35, 0.65)
                y = self.rng.uniform(-0.15, 0.15)
                ok = all(np.sqrt((x-px)**2 + (y-py)**2) > 0.07
                         for px, py in placed)
                for bp in BIN_POSITIONS.values():
                    if np.sqrt((x-bp[0])**2 + (y-bp[1])**2) < 0.12:
                        ok = False
                if ok:
                    placed.append((x, y))
                    break
            else:
                placed.append((self.rng.uniform(0.35, 0.65),
                               self.rng.uniform(-0.15, 0.15)))

            jid = self._ball_jnt_ids[i]
            adr = self.model.jnt_qposadr[jid]
            self.data.qpos[adr:adr+3] = [placed[-1][0], placed[-1][1],
                                          TABLE_Z + BALL_RADIUS + 0.001]
            self.data.qpos[adr+3:adr+7] = [1, 0, 0, 0]
            dof = self.model.jnt_dofadr[jid]
            self.data.qvel[dof:dof+6] = 0.0

    def _hide_ball(self, idx: int):
        jid = self._ball_jnt_ids[idx]
        adr = self.model.jnt_qposadr[jid]
        self.data.qpos[adr:adr+3] = [10, 10, -10.0]
        dof = self.model.jnt_dofadr[jid]
        self.data.qvel[dof:dof+6] = 0.0

    def _set_ball_colors(self):
        for i in range(self.n_balls):
            gid = self._ball_geom_ids[i]
            self.model.geom_rgba[gid] = BALL_COLORS[self.ball_colors_ep[i]].copy()

    def _get_ball_pos(self, idx: int) -> np.ndarray:
        return self.data.xpos[self._ball_body_ids[idx]].copy()

    def _get_ball_vel(self, idx: int) -> np.ndarray:
        jid = self._ball_jnt_ids[idx]
        dof = self.model.jnt_dofadr[jid]
        return self.data.qvel[dof:dof+3].copy()

    # Domain Randomization 

    def _apply_domain_rand(self):
        for lid in range(self.model.nlight):
            jitter = self.rng.uniform(0.85, 1.15, 3)
            self.model.light_diffuse[lid] = np.clip(
                self.model.light_diffuse[lid] * jitter, 0, 1)
        for i in range(self.n_balls):
            gid = self._ball_geom_ids[i]
            base = BALL_COLORS[self.ball_colors_ep[i]][:3].copy()
            noise = self.rng.uniform(-0.08, 0.08, 3)
            self.model.geom_rgba[gid, :3] = np.clip(base + noise, 0.05, 0.95)

    # ID Cache 

    def _cache_ids(self):
        self._ee_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_center_site")

        self._joint_ids = []
        self._joint_qpos = []
        self._joint_dof = []
        for i in range(1, 8):
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
            self._joint_ids.append(jid)
            self._joint_qpos.append(self.model.jnt_qposadr[jid])
            self._joint_dof.append(self.model.jnt_dofadr[jid])

        f1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
        f2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")
        self._f1_adr = self.model.jnt_qposadr[f1]
        self._f2_adr = self.model.jnt_qposadr[f2]

        self._ball_body_ids = []
        self._ball_geom_ids = []
        self._ball_jnt_ids = []
        for i in range(N_MAX_BALLS):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"ball{i}")
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"ball{i}_geom")
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"ball{i}_joint")
            self._ball_body_ids.append(bid)
            self._ball_geom_ids.append(gid)
            self._ball_jnt_ids.append(jid)

    # Render

    def render(self):
        if not hasattr(self, '_renderer'):
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera="overhead_cam")
        return self._renderer.render()

    def close(self):
        if hasattr(self, '_renderer'):
            self._renderer.close()


# Smoke test 

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("PandaSortEnv v2 — Smoke Test (multi-ball obs)")
    print("=" * 60)

    # Test 1: 1-ball
    print("\n[1] 1-ball reset")
    env = PandaSortEnv(n_balls=1, seed=42)
    obs, info = env.reset()
    print(f"  obs shape: {obs.shape} (expected {OBS_DIM})")
    print(f"  ball colors: {info['ball_colors']}")
    print(f"  ee_pos: {obs[0:3]}")
    print(f"  ball0 pos: {obs[7:10]}")
    print(f"  ball0 active: {obs[17]}")
    print(f"  ball1 active: {obs[28]}")  # slot 1 should be inactive
    assert obs.shape == (OBS_DIM,)
    assert obs[17] == 1.0   # ball0 active
    assert obs[28] == 0.0   # ball1 inactive (only 1 ball)
    print("  OK")

    # Test 2: 4-ball
    print("\n[2] 4-ball reset")
    env4 = PandaSortEnv(n_balls=4, seed=99)
    obs4, info4 = env4.reset()
    print(f"  obs shape: {obs4.shape}")
    print(f"  ball colors: {info4['ball_colors']}")
    for i in range(4):
        offset = 7 + i * PER_BALL_DIM
        pos = obs4[offset:offset+3]
        active = obs4[offset+10]
        print(f"  ball{i}: pos={pos}, active={active}")
    assert all(obs4[7 + i*PER_BALL_DIM + 10] == 1.0 for i in range(4))
    print("  OK")

    # Test 3: Scripted 1-ball pick-and-place
    print("\n[3] Scripted 1-ball pick-and-place")
    env = PandaSortEnv(n_balls=1, seed=123)
    obs, info = env.reset()

    ball_pos = obs[7:10]
    color = info['ball_colors'][0]
    bin_pos = BIN_POSITIONS[color]
    print(f"  Ball at: {ball_pos}, color: {color}, bin: {bin_pos}")

    def go_to(env, target, grip, n_steps):
        for _ in range(n_steps):
            ee = env.osc.get_ee_pos()
            delta = target - ee
            action = np.zeros(4)
            action[0:3] = np.clip(delta / POS_SCALE, -1.0, 1.0)
            action[3] = grip
            obs, r, done, trunc, info = env.step(action)
        return obs, info

    above = ball_pos.copy(); above[2] = 0.55
    at_ball = ball_pos.copy(); at_ball[2] = TABLE_Z + BALL_RADIUS
    lift = ball_pos.copy(); lift[2] = 0.55
    above_bin = bin_pos.copy(); above_bin[2] = 0.55
    at_bin = bin_pos.copy(); at_bin[2] = TABLE_Z + BALL_RADIUS + 0.02

    go_to(env, above, 1.0, 15)
    go_to(env, at_ball, 1.0, 10)
    go_to(env, at_ball, -1.0, 10)
    go_to(env, lift, -1.0, 10)
    go_to(env, above_bin, -1.0, 15)
    go_to(env, at_bin, -1.0, 10)
    go_to(env, at_bin, 1.0, 5)
    go_to(env, above_bin, 1.0, 5)

    for _ in range(5):
        obs, _, _, _, info = env.step(np.array([0, 0, 0, 1.0]))

    sorted_ok = info['n_sorted'] >= 1
    print(f"  Sorted: {sorted_ok}")
    print(f"  {'PASS' if sorted_ok else 'FAIL'}")

    # Test 4: Random steps timing
    print("\n[4] Timing (50 random steps)")
    env.reset()
    t0 = time.time()
    for _ in range(50):
        env.step(env.action_space.sample())
    elapsed = time.time() - t0
    print(f"  {elapsed/50*1000:.1f}ms/step")

    env.close()
    env4.close()
    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED" if sorted_ok else "SMOKE TEST FAILED")
    print("=" * 60)