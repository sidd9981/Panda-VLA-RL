"""
Collect demonstrations for SmolVLA training using LeRobot dataset format.

Uses TRAINED state-based policies as oracles:
    - 1-ball: SAC checkpoint (100% success)
    - 2-ball: HRL checkpoint (100% success)
    - 4-ball: HRL checkpoint (88% success)

Records only what SmolVLA will see:
    - observation.images.top: overhead camera (480x640)
    - observation.images.wrist: EE-tracking wrist camera (480x640)
    - observation.state: 6D proprio (ee_pos + ee_vel)
    - action: 6D (4D action zero-padded to 6)
    - task: natural language instruction per frame

Usage:
    # 1-ball (SAC oracle)
    python scripts/collect_smolvla_demos.py --n_episodes 100 --n_balls 1

    # 2-ball (HRL oracle)
    python scripts/collect_smolvla_demos.py --n_episodes 100 --n_balls 2

    # 4-ball (HRL oracle)
    python scripts/collect_smolvla_demos.py --n_episodes 80 --n_balls 4

    # All at once
    python scripts/collect_smolvla_demos.py --mixed
"""

import argparse
import numpy as np
import mujoco
import time
from pathlib import Path
import sys
import PIL.Image

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))
from env.panda_sort_env import (
    PandaSortEnv, OBS_DIM, PER_BALL_DIM, N_MAX_BALLS,
    TABLE_Z, BALL_RADIUS, BIN_POSITIONS, POS_SCALE, LIFT_THRESH,
)
from train_sac import SAC
from hrl_controller import (
    HierarchicalController, make_1ball_obs, get_valid_mask
)

#  Constants 

IMG_H, IMG_W = 480, 640
FPS = 10

# Default checkpoint paths
CHECKPOINTS = {
    1: "runs/sac_1ball_new/final_model.pt",
    2: "runs/hrl_2balls_new/final_model.pt",
    4: "runs/hrl_4balls_new/final_model.pt",
}


#  Camera rendering 

class DemoRenderer:
    """Renders overhead + wrist cameras at 480x640 for demo collection."""

    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.model.vis.global_.offwidth = IMG_W
        self.model.vis.global_.offheight = IMG_H

        self.overhead_renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)
        self.wrist_renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)

        self._ee_site = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "ee_center_site")

    def render(self):
        """Returns (overhead_img, wrist_img) as PIL Images."""
        self.overhead_renderer.update_scene(self.data, camera="overhead_cam")
        overhead = self.overhead_renderer.render().copy()

        ee_pos = self.data.site_xpos[self._ee_site].copy()
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = ee_pos - np.array([0, 0, 0.08])
        cam.distance = 0.20
        cam.azimuth = 180
        cam.elevation = -75
        self.wrist_renderer.update_scene(self.data, camera=cam)
        wrist = self.wrist_renderer.render().copy()

        return PIL.Image.fromarray(overhead), PIL.Image.fromarray(wrist)

    def close(self):
        self.overhead_renderer.close()
        self.wrist_renderer.close()


#  Oracle wrappers 

class SACOracle:
    """1-ball oracle using trained SAC policy."""

    def __init__(self, checkpoint_path):
        self.agent = SAC(obs_dim=OBS_DIM, act_dim=4, hidden=256, device="cpu")
        self.agent.load(checkpoint_path)
        print(f"  Loaded 1-ball SAC oracle from {checkpoint_path}")

    def reset(self):
        pass

    def select_action(self, obs):
        return self.agent.select_action(obs, deterministic=True)


class HRLOracle:
    """Multi-ball oracle using trained HRL (DQN high-level + frozen SAC low-level)."""

    def __init__(self, low_level_path, hrl_path, n_balls, max_steps_per_ball=150):
        self.low_level = SAC(obs_dim=OBS_DIM, act_dim=4, hidden=256, device="cpu")
        self.low_level.load(low_level_path)

        self.hrl = HierarchicalController(
            self.low_level, n_balls=n_balls,
            max_steps_per_ball=max_steps_per_ball,
        )
        self.hrl.load(hrl_path)
        print(f"  Loaded {n_balls}-ball HRL oracle from {hrl_path}")

    def reset(self):
        self.hrl.reset()

    def select_action(self, obs):
        # High-level picks target if needed
        if self.hrl.needs_new_target(obs):
            valid = get_valid_mask(obs)
            if not np.any(valid):
                return np.zeros(4, dtype=np.float32)
            self.hrl.select_target(obs, epsilon=0.0)

        # Low-level executes
        return self.hrl.select_action(obs)


def load_oracle(n_balls, sac_ckpt=None, hrl_ckpt=None):
    """Load the right oracle for the given ball count."""
    sac_path = sac_ckpt or CHECKPOINTS[1]

    if n_balls == 1:
        return SACOracle(sac_path)
    else:
        hrl_path = hrl_ckpt or CHECKPOINTS.get(n_balls)
        if hrl_path is None:
            raise ValueError(f"No HRL checkpoint for {n_balls} balls. "
                             f"Available: {list(CHECKPOINTS.keys())}")
        return HRLOracle(sac_path, hrl_path, n_balls)


#  Task string generation 

def generate_task_string(ball_colors, n_balls):
    """Generate natural language instruction."""
    if n_balls == 1:
        color = ball_colors[0]
        return f"pick up the {color} ball and place it in the {color} bin"
    else:
        color_list = " and ".join(
            [f"the {c} ball" for c in ball_colors[:n_balls]])
        return f"sort {color_list} into their matching bins"


#  Collection 

def collect_n_balls(n_balls, n_episodes, repo_id, seed, skip_failures,
                    sac_ckpt=None, hrl_ckpt=None, root=None):
    """Collect demos for a specific ball count."""
    np.random.seed(seed)

    env = PandaSortEnv(
        n_balls=n_balls,
        reward_mode="dense",
        max_episode_steps=n_balls * 200,
        seed=seed,
    )
    renderer = DemoRenderer(env.model, env.data)
    oracle = load_oracle(n_balls, sac_ckpt, hrl_ckpt)

    print(f"\n{'='*60}")
    print(f"Collecting {n_balls}-ball demos")
    print(f"{'='*60}")
    print(f"  N episodes:  {n_episodes}")
    print(f"  FPS:         {FPS}")
    print(f"  Image size:  {IMG_H}x{IMG_W}")
    print(f"  Output:      {repo_id}")

    #  LeRobot dataset 
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        features = {
            "observation.images.top": {
                "dtype": "video",
                "shape": (3, IMG_H, IMG_W),
                "names": ["channels", "height", "width"],
            },
            "observation.images.wrist": {
                "dtype": "video",
                "shape": (3, IMG_H, IMG_W),
                "names": ["channels", "height", "width"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["ee_x", "ee_y", "ee_z", "ee_vx", "ee_vy", "ee_vz"],
            },
            "action": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["dx", "dy", "dz", "gripper", "pad1", "pad2"],
            },
        }

        root_path = Path(root) if root else None
        # Clean up existing dataset dir so LeRobot.create() doesn't choke
        if root_path is not None:
            ds_path = root_path / repo_id
            if ds_path.exists():
                import shutil
                shutil.rmtree(ds_path)
                print(f"  Removed existing dataset at {ds_path}")
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=FPS,
            features=features,
            root=root_path,
        )
        use_lerobot = True
        print(f"  Using LeRobot API (v3 format)")

    except Exception as e:
        print(f"  LeRobot API failed: {e}")
        use_lerobot = False

    #  Collection loop 
    successes = 0
    total_frames = 0
    saved_episodes = 0
    t_start = time.time()

    for ep in range(n_episodes):
        obs, info = env.reset()
        oracle.reset()
        ball_colors = info["ball_colors"]
        task = generate_task_string(ball_colors, n_balls)

        ep_frames = []
        for t in range(env.max_episode_steps):
            # Oracle selects action from state obs
            action = oracle.select_action(obs)

            # Step env
            next_obs, reward, done, trunc, info = env.step(action)

            # Record what SmolVLA will see
            overhead_img, wrist_img = renderer.render()
            proprio = obs[0:6].astype(np.float32)  # ee_pos + ee_vel
            action_6d = np.zeros(6, dtype=np.float32)
            action_6d[0:4] = action

            frame = {
                "observation.images.top": overhead_img,
                "observation.images.wrist": wrist_img,
                "observation.state": proprio,
                "action": action_6d,
                "task": task,
            }
            ep_frames.append(frame)

            obs = next_obs
            if done or trunc:
                break

        # Check success
        n_sorted = info.get("n_sorted", 0)
        success = n_sorted >= n_balls
        successes += int(success)

        status = "OK" if success else "FAIL"
        print(f"  Ep {ep+1:3d}/{n_episodes}: "
              f"{[str(c) for c in ball_colors[:n_balls]]} -> {status} "
              f"({len(ep_frames)} frames)")

        if not success and skip_failures:
            continue

        # Save to dataset
        total_frames += len(ep_frames)
        saved_episodes += 1

        if use_lerobot:
            for frame in ep_frames:
                dataset.add_frame(frame)
            dataset.save_episode()

    #  Finalize 
    if use_lerobot and saved_episodes > 0:
        print(f"\n  Finalizing dataset (encoding videos, computing stats)...")
        dataset.finalize()
        print(f"  Dataset saved to: {dataset.root}")

    elapsed = time.time() - t_start
    print(f"\n  Results: {successes}/{n_episodes} success "
          f"({successes/n_episodes:.0%}), "
          f"{saved_episodes} saved, {total_frames} frames, "
          f"{elapsed:.0f}s")

    renderer.close()
    env.close()

    return saved_episodes, total_frames


#  Main 

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_episodes", type=int, default=100)
    p.add_argument("--n_balls", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repo_id", type=str, default=None)
    p.add_argument("--root", type=str, default=None)
    p.add_argument("--skip_failures", action="store_true")
    p.add_argument("--sac_ckpt", type=str, default=None,
                   help="Override 1-ball SAC checkpoint path")
    p.add_argument("--hrl_ckpt", type=str, default=None,
                   help="Override HRL checkpoint path")

    # Mixed collection mode
    p.add_argument("--mixed", action="store_true",
                   help="Collect 1-ball + 2-ball + 4-ball in one run")

    cfg = p.parse_args()

    print("=" * 60)
    print("SmolVLA Demo Collection (Trained Oracle)")
    print("=" * 60)

    if cfg.mixed:
        # Collect all ball counts — each gets its own root to avoid conflicts
        runs = [
            (1, 1000, "demos", "1_ball_demos_v2", 108),
            (2, 1000, "demos", "2_ball_demos_v2", 18),
            (4, 1000,  "demos", "4_ball_demos_v2", 189),
        ]

        total_eps = 0
        total_fr = 0
        for n_balls, n_eps, repo_id, root, seed in runs:
            eps, fr = collect_n_balls(
                n_balls=n_balls,
                n_episodes=n_eps,
                repo_id=repo_id,
                seed=seed,
                skip_failures=True,
                sac_ckpt=cfg.sac_ckpt,
                hrl_ckpt=None,
                root=root,
            )
            total_eps += eps
            total_fr += fr

        print(f"\n{'='*60}")
        print(f"ALL COLLECTIONS COMPLETE")
        print(f"{'='*60}")
        print(f"  Total episodes: {total_eps}")
        print(f"  Total frames:   {total_fr}")
        print(f"\n  Next: merge datasets with lerobot-edit-dataset")

    else:
        # Single ball count
        repo_id = cfg.repo_id or f"local/panda_sort_{cfg.n_balls}ball"
        collect_n_balls(
            n_balls=cfg.n_balls,
            n_episodes=cfg.n_episodes,
            repo_id=repo_id,
            seed=cfg.seed,
            skip_failures=cfg.skip_failures,
            sac_ckpt=cfg.sac_ckpt,
            hrl_ckpt=cfg.hrl_ckpt,
            root=cfg.root,
        )


if __name__ == "__main__":
    main()