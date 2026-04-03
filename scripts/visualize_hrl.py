"""
Visualize trained HRL policy sorting multiple balls.

Records overhead + side + front MP4s showing the full hierarchical system:
    - High-level DQN selects which ball to target
    - Low-level frozen SAC executes pick-and-place

Usage:
    python scripts/visualize_hrl.py \
        --low_level runs/sac_1ball_2/best_model.pt \
        --hl_checkpoint runs/hrl_2ball_seed0/best_model.pt \
        --n_balls 2
"""

import argparse
import numpy as np
import torch
import mujoco
import cv2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))
from env.panda_sort_env import PandaSortEnv, OBS_DIM, PER_BALL_DIM, N_MAX_BALLS
from scripts.train_sac import SAC
from hrl_controller import HierarchicalController, get_valid_mask


def run_hrl_episode(env, hrl, max_steps=None):
    """Run one episode, return (obs_list, info_list, actions, hl_decisions)."""
    if max_steps is None:
        max_steps = env.max_episode_steps

    obs, info = env.reset()
    hrl.reset()

    trajectory = []
    hl_decisions = []

    for t in range(max_steps):
        if hrl.needs_new_target(obs):
            valid = get_valid_mask(obs)
            if not np.any(valid):
                break
            target = hrl.select_target(obs, epsilon=0.0)
            hl_decisions.append((t, target))

        action = hrl.select_action(obs)
        obs, reward, done, trunc, info = env.step(action)
        trajectory.append((obs.copy(), info.copy(), action.copy()))

        if done or trunc:
            break

    return trajectory, hl_decisions, info


def save_video(env, hrl, video_path, camera, n_episodes=3, fps=30):
    """Record episodes from a given camera."""
    env.model.vis.global_.offwidth = 960
    env.model.vis.global_.offheight = 720
    renderer = mujoco.Renderer(env.model, height=720, width=960)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (960, 720))

    for ep in range(n_episodes):
        obs, info = env.reset()
        hrl.reset()
        colors = info['ball_colors']
        print(f"    Ep {ep+1}: balls={colors}", end="")

        for step in range(env.max_episode_steps):
            if hrl.needs_new_target(obs):
                valid = get_valid_mask(obs)
                if not np.any(valid):
                    break
                hrl.select_target(obs, epsilon=0.0)

            action = hrl.select_action(obs)
            obs, reward, done, trunc, info = env.step(action)

            if isinstance(camera, str):
                renderer.update_scene(env.data, camera=camera)
            else:
                renderer.update_scene(env.data, camera=camera)
            frame = renderer.render()
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            if done or trunc:
                for _ in range(int(fps * 1.5)):
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                break

        n_sorted = info.get("n_sorted", 0)
        status = "OK" if n_sorted >= env.n_balls else "!"
        print(f" -> {step+1} steps, sorted {n_sorted}/{env.n_balls} {status}")

    writer.release()
    renderer.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--low_level", type=str, required=True,
                   help="Path to frozen 1-ball SAC checkpoint")
    p.add_argument("--hl_checkpoint", type=str, required=True,
                   help="Path to trained HRL high-level checkpoint")
    p.add_argument("--n_balls", type=int, default=2)
    p.add_argument("--n_episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default=None)
    args = p.parse_args()

    # Output dir
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(args.hl_checkpoint).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load low-level
    print("Loading low-level SAC...")
    low_level = SAC(obs_dim=OBS_DIM, act_dim=4, hidden=256, device="cpu")
    low_level.load(args.low_level)

    # Build HRL controller
    hrl = HierarchicalController(
        low_level, n_balls=args.n_balls, max_steps_per_ball=150
    )
    hrl.load(args.hl_checkpoint)
    print(f"Loaded HRL from {args.hl_checkpoint}")

    # Env
    env = PandaSortEnv(
        n_balls=args.n_balls,
        max_episode_steps=args.n_balls * 150 + 50,
        seed=args.seed,
    )

    #  Diagnostic 
    print(f"\n{'='*50}")
    print(f"DIAGNOSTIC — {args.n_balls} balls, {args.n_episodes} episodes")
    print(f"{'='*50}")

    successes = 0
    for ep in range(args.n_episodes):
        traj, hl_decs, info = run_hrl_episode(env, hrl)
        n_sorted = info.get("n_sorted", 0)
        ok = n_sorted >= env.n_balls
        successes += int(ok)
        steps = len(traj)
        hl_str = ", ".join(f"t={t}->ball{b}" for t, b in hl_decs)
        print(f"  Ep {ep+1}: {'✓' if ok else '✗'} "
              f"sorted={n_sorted}/{env.n_balls} "
              f"steps={steps} HL=[{hl_str}]")

    print(f"\n  Success: {successes}/{args.n_episodes} "
          f"({successes/args.n_episodes:.0%})")

    #  Overhead video 
    overhead_path = out_dir / f"hrl_{args.n_balls}ball_overhead.mp4"
    print(f"\n  Recording overhead -> {overhead_path}")
    save_video(env, hrl, str(overhead_path), "overhead_cam", args.n_episodes)

    #  Side video 
    side_path = out_dir / f"hrl_{args.n_balls}ball_side.mp4"
    print(f"\n  Recording side -> {side_path}")

    side_cam = mujoco.MjvCamera()
    side_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    side_cam.lookat[:] = [0.55, 0.0, 0.45]
    side_cam.distance = 1.2
    side_cam.azimuth = 160
    side_cam.elevation = -25

    save_video(env, hrl, str(side_path), side_cam, args.n_episodes)

    #  Front video 
    front_path = out_dir / f"hrl_{args.n_balls}ball_front.mp4"
    print(f"\n  Recording front -> {front_path}")

    front_cam = mujoco.MjvCamera()
    front_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    front_cam.lookat[:] = [0.55, 0.0, 0.45]
    front_cam.distance = 1.0
    front_cam.azimuth = 180
    front_cam.elevation = -15

    save_video(env, hrl, str(front_path), front_cam, args.n_episodes)

    env.close()
    print(f"\n  All videos saved to {out_dir}/")
    print("  Done.")


if __name__ == "__main__":
    main()