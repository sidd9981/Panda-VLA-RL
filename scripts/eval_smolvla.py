"""
Evaluate fine-tuned SmolVLA on PandaSortEnv.

Loads the trained SmolVLA checkpoint, runs it in the sim with
camera rendering, and measures success rates across ball counts
and language instructions.

Tests:
    1. Standard eval: same task distribution as training
    2. Novel instructions: unseen language phrasings
    3. Generalization: unseen ball positions, counts

Usage:
    python scripts/eval_smolvla.py
    python scripts/eval_smolvla.py --checkpoint outputs/train/smolvla_panda_sort_v2/checkpoints/020000/pretrained_model
    python scripts/eval_smolvla.py --n_balls 2 --n_episodes 50
    python scripts/eval_smolvla.py --novel_instructions
"""

import argparse
import numpy as np
import torch
import mujoco
import time
from pathlib import Path
import sys
import PIL.Image

sys.path.append(str(Path(__file__).parent.parent))
from env.panda_sort_env import (
    PandaSortEnv, OBS_DIM, PER_BALL_DIM, N_MAX_BALLS,
    TABLE_Z, BALL_RADIUS, BIN_POSITIONS, POS_SCALE,
)

#  Constants 

IMG_H, IMG_W = 480, 640
FPS = 10


#  Camera rendering (same as demo collection) 

class EvalRenderer:
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


#  SmolVLA inference wrapper 

class SmolVLAAgent:
    """Wraps SmolVLA for inference in the sim environment."""

    def __init__(self, checkpoint_path, device="mps"):
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self.device = torch.device(device)

        print(f"Loading SmolVLA from {checkpoint_path}...")
        self.policy = SmolVLAPolicy.from_pretrained(
            checkpoint_path
        ).to(self.device).eval()

        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            checkpoint_path,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

        # Action chunk buffer
        self._action_buffer = []
        self._action_idx = 0
        print(f"  Loaded. chunk_size={self.policy.config.chunk_size}, "
              f"n_action_steps={self.policy.config.n_action_steps}")

    @torch.no_grad()
    def predict(self, overhead_img, wrist_img, state, task_str):
        """
        Get next action from SmolVLA.

        Uses action chunking: predicts chunk_size actions at once,
        then executes them one at a time. Only re-predicts when
        the buffer is exhausted.
        """
        if self._action_idx < len(self._action_buffer):
            action = self._action_buffer[self._action_idx]
            self._action_idx += 1
            return action

        # Build batch dict matching SmolVLA's expected format
        oh_tensor = torch.from_numpy(
            np.array(overhead_img).transpose(2, 0, 1).astype(np.float32) / 255.0
        )
        wr_tensor = torch.from_numpy(
            np.array(wrist_img).transpose(2, 0, 1).astype(np.float32) / 255.0
        )

        batch = {
            "observation.images.camera1": oh_tensor,
            "observation.images.camera2": wr_tensor,
            "observation.state": torch.from_numpy(state).float(),
            "task": task_str,
        }

        batch = self.preprocess(batch)
        output = self.policy.select_action(batch)
        output = self.postprocess(output)

        if isinstance(output, dict):
            actions = output["action"].cpu().numpy()
        elif isinstance(output, torch.Tensor):
            actions = output.cpu().numpy()
        else:
            actions = np.array(output)
        if actions.ndim == 1:
            actions = actions[np.newaxis, :]

        n_action_steps = min(
            self.policy.config.n_action_steps,
            len(actions)
        )
        self._action_buffer = [actions[i] for i in range(n_action_steps)]
        self._action_idx = 1

        return self._action_buffer[0]

    def reset(self):
        """Clear action buffer for new episode."""
        self._action_buffer = []
        self._action_idx = 0


#  Task string generators 

def standard_task_string(ball_colors, n_balls):
    """Same format as training data."""
    if n_balls == 1:
        color = ball_colors[0]
        return f"pick up the {color} ball and place it in the {color} bin"
    else:
        color_list = " and ".join(
            [f"the {c} ball" for c in ball_colors[:n_balls]])
        return f"sort {color_list} into their matching bins"


def novel_task_strings(ball_colors, n_balls):
    """Unseen phrasings to test language generalization."""
    rng = np.random.default_rng()

    if n_balls == 1:
        color = ball_colors[0]
        options = [
            f"grab the {color} ball and drop it in the {color} bin",
            f"move the {color} sphere to its matching container",
            f"put the {color} ball into the {color} box",
            f"place {color} in the {color} bin",
            f"take the {color} one and sort it",
        ]
    else:
        options = [
            "sort all the balls into their matching bins",
            "put each ball in the bin that matches its color",
            "organize the colored balls by placing them in the correct bins",
            "clean up the table by sorting balls into bins",
            "move every ball to its corresponding colored bin",
        ]

    return rng.choice(options)


#  Evaluation 

def evaluate(agent, n_balls, n_episodes, seed, use_novel_instructions=False):
    """Run evaluation episodes and collect metrics."""

    env = PandaSortEnv(
        n_balls=n_balls,
        reward_mode="dense",
        max_episode_steps=n_balls * 200,
        seed=seed,
    )
    renderer = EvalRenderer(env.model, env.data)

    results = {
        "success": 0,
        "n_sorted_total": 0,
        "n_balls_total": 0,
        "ep_lengths": [],
        "rewards": [],
        "reached": 0,
        "grasped": 0,
        "lifted": 0,
    }

    for ep in range(n_episodes):
        obs, info = env.reset()
        agent.reset()
        ball_colors = info["ball_colors"]

        if use_novel_instructions:
            task = novel_task_strings(ball_colors, n_balls)
        else:
            task = standard_task_string(ball_colors, n_balls)

        ep_reward = 0
        reached = grasped = lifted = False

        for t in range(env.max_episode_steps):
            overhead_img, wrist_img = renderer.render()
            state = obs[0:6].astype(np.float32)

            action_6d = agent.predict(overhead_img, wrist_img, state, task)

            # Extract 4D action (ignore zero-padded dims)
            action_4d = action_6d[:4] if len(action_6d) >= 4 else np.zeros(4)
            action_4d = np.clip(action_4d, -1.0, 1.0)

            obs, reward, done, trunc, info = env.step(action_4d)
            ep_reward += reward

            # Track sub-task metrics
            fw = obs[6]
            for i in range(n_balls):
                offset = 7 + i * PER_BALL_DIM
                ball_rel = obs[offset + 3:offset + 6]
                ball_pos = obs[offset:offset + 3]
                dist = np.linalg.norm(ball_rel)
                if dist < 0.05:
                    reached = True
                if fw > 0.01 and fw < 0.06 and dist < 0.04:
                    grasped = True
                if ball_pos[2] > TABLE_Z + BALL_RADIUS + 0.08:
                    lifted = True

            if done or trunc:
                break

        n_sorted = info.get("n_sorted", 0)
        success = n_sorted >= n_balls
        results["success"] += int(success)
        results["n_sorted_total"] += n_sorted
        results["n_balls_total"] += n_balls
        results["ep_lengths"].append(t + 1)
        results["rewards"].append(ep_reward)
        results["reached"] += int(reached)
        results["grasped"] += int(grasped)
        results["lifted"] += int(lifted)

        status = "OK" if success else "FAIL"
        max_h = max(info.get("max_heights", [0]))
        print(f"  Ep {ep+1:3d}/{n_episodes}: {[str(c) for c in ball_colors[:n_balls]]} "
              f"-> {status} sorted={n_sorted}/{n_balls} steps={t+1} "
              f"max_z={max_h:.3f} "
              f"reach={'Y' if reached else 'N'} "
              f"grasp={'Y' if grasped else 'N'} "
              f"lift={'Y' if lifted else 'N'}")

    renderer.close()
    env.close()

    n = n_episodes
    return {
        "success_rate": results["success"] / n,
        "sort_rate": results["n_sorted_total"] / results["n_balls_total"],
        "reach_rate": results["reached"] / n,
        "grasp_rate": results["grasped"] / n,
        "lift_rate": results["lifted"] / n,
        "avg_steps": np.mean(results["ep_lengths"]),
        "avg_reward": np.mean(results["rewards"]),
        "n_episodes": n,
    }


#  Main 

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,
                   default="outputs/train/smolvla_panda_sort_v2/checkpoints/020000/pretrained_model",
                   help="Path to trained SmolVLA checkpoint")
    p.add_argument("--n_balls", type=int, default=None,
                   help="Specific ball count to test (default: test all)")
    p.add_argument("--n_episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--novel_instructions", action="store_true",
                   help="Test with unseen language phrasings")
    cfg = p.parse_args()

    print("=" * 60)
    print("SmolVLA Evaluation — PandaSortEnv")
    print("=" * 60)
    print(f"  Checkpoint: {cfg.checkpoint}")
    print(f"  Device:     {cfg.device}")
    print(f"  Novel lang: {cfg.novel_instructions}")

    agent = SmolVLAAgent(cfg.checkpoint, device=cfg.device)

    if cfg.n_balls is not None:
        ball_counts = [cfg.n_balls]
    else:
        ball_counts = [1, 2, 4]

    all_results = {}
    for n_balls in ball_counts:
        print(f"\n{'='*50}")
        print(f"  {n_balls}-BALL EVALUATION ({cfg.n_episodes} episodes)")
        print(f"{'='*50}")

        rates = evaluate(
            agent, n_balls, cfg.n_episodes, cfg.seed,
            use_novel_instructions=cfg.novel_instructions,
        )
        all_results[n_balls] = rates

        print(f"\n  Reach:      {rates['reach_rate']:.0%}")
        print(f"  Grasp:      {rates['grasp_rate']:.0%}")
        print(f"  Lift:       {rates['lift_rate']:.0%}")
        print(f"  Success:    {rates['success_rate']:.0%}")
        print(f"  Sort rate:  {rates['sort_rate']:.0%}")
        print(f"  Avg steps:  {rates['avg_steps']:.0f}")
        print(f"  Avg reward: {rates['avg_reward']:.1f}")

    #  Summary 
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    lang = "novel" if cfg.novel_instructions else "standard"
    print(f"  Language: {lang}")
    print(f"  {'Balls':<8} {'Reach':<8} {'Grasp':<8} {'Lift':<8} {'Place':<8} {'Sort%':<8}")
    print(f"  {'-'*48}")
    for n_balls, rates in all_results.items():
        print(f"  {n_balls:<8} {rates['reach_rate']:<8.0%} "
              f"{rates['grasp_rate']:<8.0%} {rates['lift_rate']:<8.0%} "
              f"{rates['success_rate']:<8.0%} {rates['sort_rate']:<8.0%}")

    #  Compare with baselines 
    print(f"\n  Baselines:")
    print(f"  {'Method':<25} {'1-ball':<10} {'2-ball':<10} {'4-ball':<10}")
    print(f"  {'-'*55}")
    print(f"  {'State SAC':<25} {'100%':<10} {'—':<10} {'—':<10}")
    print(f"  {'State HRL':<25} {'—':<10} {'100%':<10} {'88%':<10}")
    print(f"  {'Vision SAC':<25} {'100%':<10} {'—':<10} {'—':<10}")
    smolvla_row = "  SmolVLA BC             "
    for nb in [1, 2, 4]:
        if nb in all_results:
            smolvla_row += f"{all_results[nb]['success_rate']:<10.0%}"
        else:
            smolvla_row += f"{'—':<10}"
    print(smolvla_row)


if __name__ == "__main__":
    main()