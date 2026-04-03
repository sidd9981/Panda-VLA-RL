"""
Save videos of trained SAC policy on PandaSortEnv v2.

Saves three MP4s:
    - overhead view (sees full table, bins, arm from above)
    - side view (sees lift height, drop clearly)
    - front view (sees the drop clearly)

Usage:
    python scripts/visualize.py --checkpoint runs/sac_1ball/best_model.pt
    python scripts/visualize.py --checkpoint runs/sac_1ball/best_model.pt --n_episodes 10
"""

import argparse
import numpy as np
import torch
import mujoco
import cv2
import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))
from env.panda_sort_env import PandaSortEnv, OBS_DIM, TABLE_Z, BIN_POSITIONS
from scripts.train_sac import SAC


def save_video(env, agent, video_path, camera, n_episodes=3, fps=30):
    """Render episodes from a given camera and save as MP4."""
    env.model.vis.global_.offwidth = 960
    env.model.vis.global_.offheight = 720
    renderer = mujoco.Renderer(env.model, height=720, width=960)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (960, 720))

    for ep in range(n_episodes):
        obs, info = env.reset()
        colors = info['ball_colors']
        print(f"    Ep {ep+1}: balls={colors}", end="")

        for step in range(env.max_episode_steps):
            action = agent.select_action(obs, deterministic=True)
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
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--n_episodes", type=int, default=5)
    p.add_argument("--n_balls",    type=int, default=1)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--output_dir", type=str, default=None)
    args = p.parse_args()

    # Output dir
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(args.checkpoint).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load agent
    print("Loading agent...")
    agent = SAC(obs_dim=OBS_DIM, act_dim=4, hidden=256, device="cpu")
    agent.load(args.checkpoint)

    env = PandaSortEnv(n_balls=args.n_balls, seed=args.seed)

    #  Diagnostic run 
    print(f"\n{'='*50}")
    print(f"DIAGNOSTIC — {args.n_balls} ball(s), {args.n_episodes} episodes")
    print(f"{'='*50}")
    successes = 0
    for ep in range(args.n_episodes):
        obs, info = env.reset()
        colors = info['ball_colors']
        for step in range(env.max_episode_steps):
            action = agent.select_action(obs, deterministic=True)
            obs, r, done, trunc, info = env.step(action)
            if done or trunc:
                break
        n_sorted = info.get("n_sorted", 0)
        ok = n_sorted >= env.n_balls
        successes += int(ok)
        max_h = max(info.get("max_heights", [0]))
        print(f"  Ep {ep+1}: {colors} -> {'OK' if ok else '!'} "
              f"sorted={n_sorted}/{env.n_balls} steps={step+1} max_z={max_h:.3f}")
    print(f"\n  Success: {successes}/{args.n_episodes} "
          f"({successes/args.n_episodes:.0%})")

    #  Overhead video 
    overhead_path = out_dir / "demo_overhead.mp4"
    print(f"\n  Recording overhead -> {overhead_path}")
    save_video(env, agent, str(overhead_path), "overhead_cam", args.n_episodes)

    #  Side video 
    side_path = out_dir / "demo_side.mp4"
    print(f"\n  Recording side -> {side_path}")

    side_cam = mujoco.MjvCamera()
    side_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    side_cam.lookat[:] = [0.55, 0.0, 0.45]
    side_cam.distance = 1.2
    side_cam.azimuth = 160
    side_cam.elevation = -25

    save_video(env, agent, str(side_path), side_cam, args.n_episodes)

    #  Front video 
    front_path = out_dir / "demo_front.mp4"
    print(f"\n  Recording front -> {front_path}")

    front_cam = mujoco.MjvCamera()
    front_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    front_cam.lookat[:] = [0.55, 0.0, 0.45]
    front_cam.distance = 1.0
    front_cam.azimuth = 180
    front_cam.elevation = -15

    save_video(env, agent, str(front_path), front_cam, args.n_episodes)

    env.close()
    print(f"\n  All videos saved to {out_dir}/")
    print("  Done.")


if __name__ == "__main__":
    main()