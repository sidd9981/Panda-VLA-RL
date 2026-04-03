"""
Visualize trained Vision SAC policy.

Records three MP4s:
    - overhead view
    - side view
    - wrist view (what the agent actually sees)

Also saves a side-by-side composite: overhead | wrist | side
so you can see the agent's POV alongside the task view.

Usage:
    python scripts/visualize_vision.py --checkpoint runs/vision_sac_seed0/best_model.pt
    python scripts/visualize_vision.py --checkpoint runs/vision_sac_seed0/best_model.pt --n_episodes 10
"""

import argparse
import numpy as np
import torch
import mujoco
import cv2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from env.panda_sort_env import PandaSortEnv, OBS_DIM, PER_BALL_DIM, TABLE_Z, BALL_RADIUS, LIFT_THRESH
from train_vision_sac import VisionSAC, ImageRenderer, get_proprio, RENDER_SIZE, CROP_SIZE


def make_side_camera():
    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.55, 0.0, 0.45]
    cam.distance  = 1.2
    cam.azimuth   = 160
    cam.elevation = -25
    return cam


def make_front_camera():
    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.55, 0.0, 0.45]
    cam.distance  = 1.0
    cam.azimuth   = 180
    cam.elevation = -15
    return cam


def render_wrist_large(model, data, ee_site_id, size=480):
    """Render wrist cam at larger size for visualization."""
    renderer = mujoco.Renderer(model, height=size, width=size)
    ee_pos = data.site_xpos[ee_site_id].copy()
    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = ee_pos - np.array([0, 0, 0.08])
    cam.distance  = 0.20
    cam.azimuth   = 180
    cam.elevation = -75
    renderer.update_scene(data, camera=cam)
    frame = renderer.render()
    renderer.close()
    return frame


def save_video(env, agent, renderer, video_path, camera, n_episodes, fps=30,
               width=960, height=720):
    """Record episodes from a single camera."""
    env.model.vis.global_.offwidth  = width
    env.model.vis.global_.offheight = height
    vis_renderer = mujoco.Renderer(env.model, height=height, width=width)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for ep in range(n_episodes):
        obs, info = env.reset()
        overhead, wrist = renderer.render()
        proprio = get_proprio(obs)
        colors = info['ball_colors']
        print(f"    Ep {ep+1}: {colors}", end="")

        for step in range(env.max_episode_steps):
            action = agent.select_action(overhead, wrist, proprio, deterministic=True)
            obs, r, done, trunc, info = env.step(action)
            overhead, wrist = renderer.render()
            proprio = get_proprio(obs)

            if isinstance(camera, str):
                vis_renderer.update_scene(env.data, camera=camera)
            else:
                vis_renderer.update_scene(env.data, camera=camera)
            frame = vis_renderer.render()
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            if done or trunc:
                for _ in range(int(fps * 1.5)):
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                break

        n_sorted = info.get("n_sorted", 0)
        print(f" -> {step+1} steps, sorted {n_sorted}/{env.n_balls} "
              f"{'OK' if n_sorted >= env.n_balls else '!'}")

    writer.release()
    vis_renderer.close()
    # Reset framebuffer back to RENDER_SIZE for the renderer used by the agent
    env.model.vis.global_.offwidth  = RENDER_SIZE
    env.model.vis.global_.offheight = RENDER_SIZE


def save_composite_video(env, agent, renderer, video_path, n_episodes, fps=30):
    """
    Save side-by-side composite: overhead (480x640) | wrist (480x480) | side (480x640)
    Total: 480 x 1760
    """
    W, H = 480, 480
    oh_W, oh_H = 640, 480
    side_W, side_H = 640, 480
    total_W = oh_W + W + side_W  # 1760
    total_H = 480

    # Must set framebuffer before creating renderers
    env.model.vis.global_.offwidth  = max(oh_W, side_W, W)
    env.model.vis.global_.offheight = max(oh_H, side_H, H)

    overhead_vis = mujoco.Renderer(env.model, height=oh_H, width=oh_W)
    side_vis     = mujoco.Renderer(env.model, height=side_H, width=side_W)
    wrist_vis    = mujoco.Renderer(env.model, height=H, width=W)

    side_cam = make_side_camera()
    ee_site  = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "ee_center_site")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (total_W, total_H))

    for ep in range(n_episodes):
        obs, info = env.reset()
        overhead, wrist = renderer.render()
        proprio = get_proprio(obs)
        colors = info['ball_colors']
        print(f"    Composite ep {ep+1}: {colors}", end="")

        for step in range(env.max_episode_steps):
            action = agent.select_action(overhead, wrist, proprio, deterministic=True)
            obs, r, done, trunc, info = env.step(action)
            overhead, wrist = renderer.render()
            proprio = get_proprio(obs)

            # Overhead
            overhead_vis.update_scene(env.data, camera="overhead_cam")
            oh_frame = cv2.cvtColor(overhead_vis.render(), cv2.COLOR_RGB2BGR)

            # Wrist (agent's POV — upscaled from 68px to 480px)
            ee_pos = env.data.site_xpos[ee_site].copy()
            wrist_cam = mujoco.MjvCamera()
            wrist_cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
            wrist_cam.lookat[:] = ee_pos - np.array([0, 0, 0.08])
            wrist_cam.distance  = 0.20
            wrist_cam.azimuth   = 180
            wrist_cam.elevation = -75
            wrist_vis.update_scene(env.data, camera=wrist_cam)
            wr_frame = cv2.cvtColor(wrist_vis.render(), cv2.COLOR_RGB2BGR)

            # Side
            side_vis.update_scene(env.data, camera=side_cam)
            side_frame = cv2.cvtColor(side_vis.render(), cv2.COLOR_RGB2BGR)

            # Add labels
            cv2.putText(oh_frame,   "Overhead",      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            cv2.putText(wr_frame,   "Wrist (agent)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            cv2.putText(side_frame, "Side",          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

            # Step / sorted info on overhead
            n_sorted = info.get("n_sorted", 0)
            cv2.putText(oh_frame, f"Step {step+1}  Sorted {n_sorted}/{env.n_balls}",
                        (10, oh_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            composite = np.concatenate([oh_frame, wr_frame, side_frame], axis=1)
            writer.write(composite)

            if done or trunc:
                for _ in range(int(fps * 1.5)):
                    writer.write(composite)
                break

        n_sorted = info.get("n_sorted", 0)
        print(f" -> {step+1} steps, sorted {n_sorted}/{env.n_balls} "
              f"{'OK' if n_sorted >= env.n_balls else '!'}")

    writer.release()
    overhead_vis.close()
    side_vis.close()
    wrist_vis.close()
    # Reset framebuffer back for agent renderer
    env.model.vis.global_.offwidth  = RENDER_SIZE
    env.model.vis.global_.offheight = RENDER_SIZE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--n_episodes",  type=int, default=5)
    p.add_argument("--n_balls",     type=int, default=1)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--output_dir",  type=str, default=None)
    p.add_argument("--composite_only", action="store_true",
                   help="Only save composite video, skip individual views")
    args = p.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading vision SAC agent...")
    agent = VisionSAC(per_cam_dim=128, proprio_dim=7, act_dim=4,
                      hidden=256, device="cpu")
    agent.load(args.checkpoint)

    env = PandaSortEnv(n_balls=args.n_balls, seed=args.seed)
    renderer = ImageRenderer(env.model, env.data, img_size=RENDER_SIZE)

    #  Diagnostic 
    print(f"\n{'='*50}")
    print(f"DIAGNOSTIC — {args.n_balls} ball(s), {args.n_episodes} episodes")
    print(f"{'='*50}")
    successes = 0
    for ep in range(args.n_episodes):
        obs, info = env.reset()
        overhead, wrist = renderer.render()
        proprio = get_proprio(obs)
        colors = info['ball_colors']
        for step in range(env.max_episode_steps):
            action = agent.select_action(overhead, wrist, proprio, deterministic=True)
            obs, r, done, trunc, info = env.step(action)
            overhead, wrist = renderer.render()
            proprio = get_proprio(obs)
            if done or trunc:
                break
        n_sorted = info.get("n_sorted", 0)
        ok = n_sorted >= env.n_balls
        successes += int(ok)
        max_h = max(info.get("max_heights", [0]))
        print(f"  Ep {ep+1}: {colors} -> {'OK' if ok else '!'} "
              f"sorted={n_sorted}/{env.n_balls} steps={step+1} max_z={max_h:.3f}")
    print(f"\n  Success: {successes}/{args.n_episodes} "
          f"({successes/args.n_episodes:.0%})\n")

    #  Videos 
    if not args.composite_only:
        overhead_path = out_dir / "vision_overhead.mp4"
        print(f"  Recording overhead -> {overhead_path}")
        save_video(env, agent, renderer, overhead_path,
                   "overhead_cam", args.n_episodes)

        side_path = out_dir / "vision_side.mp4"
        print(f"\n  Recording side -> {side_path}")
        save_video(env, agent, renderer, side_path,
                   make_side_camera(), args.n_episodes)

        front_path = out_dir / "vision_front.mp4"
        print(f"\n  Recording front -> {front_path}")
        save_video(env, agent, renderer, front_path,
                   make_front_camera(), args.n_episodes)

    composite_path = out_dir / "vision_composite.mp4"
    print(f"\n  Recording composite -> {composite_path}")
    save_composite_video(env, agent, renderer, composite_path, args.n_episodes)

    renderer.close()
    env.close()
    print(f"\n  Done. Videos saved to {out_dir}/")


if __name__ == "__main__":
    main()