"""
Record videos of trained VLM+SAC policy on PandaSortEnv.

Saves three MP4s per mode:
    - overhead view
    - side view
    - front view

Supports two modes:
    - 1-ball:    standard single ball sorting
    - 2-ball:    with language re-prompting after first ball sorted
                 (completion detected via vision-based color mask)

Usage:
    # 1-ball
    python scripts/visualize_vlm.py \
        --checkpoint runs/vlm_sac_1ball_seed108/best_model.pt \
        --n_balls 1

    # 2-ball with re-prompting
    python scripts/visualize_vlm.py \
        --checkpoint runs/vlm_sac_1ball_seed108/best_model.pt \
        --n_balls 2 \
        --reprompt
"""

import argparse
import numpy as np
import torch
import mujoco
import cv2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from env.panda_sort_env import (
    PandaSortEnv, OBS_DIM, PER_BALL_DIM, N_MAX_BALLS,
    TABLE_Z, BALL_RADIUS, LIFT_THRESH,
)
from scripts.train_vlm_sac import (
    VLM_SAC, VLMFeatureExtractor, ImageRenderer,
    generate_task_string, FEAT_DIM,
)


#  Vision-based completion detector 

def detect_ball_sorted_from_image(overhead_chw, color,
                                  table_thresh=30, bin_thresh=20):
    """
    Detect if a ball has been sorted using overhead camera image only.
    No state observations used.

    Checks:
        - Target color pixels in TABLE region have dropped (ball left table)
        - Target color pixels in BIN region have increased (ball arrived in bin)

    overhead_chw: (3, H, W) float32 [0, 1]
    color: "red" or "blue"
    Returns: True if ball appears to be sorted into its bin
    """
    img = overhead_chw.transpose(1, 2, 0)  # HWC
    H, W = img.shape[:2]

    if color == "red":
        mask = (img[:, :, 0] > 0.55) & (img[:, :, 1] < 0.35) & (img[:, :, 2] < 0.35)
    else:
        mask = (img[:, :, 0] < 0.35) & (img[:, :, 1] < 0.35) & (img[:, :, 2] > 0.55)

    # Overhead cam: pos="0.5 0.0 1.2", fovy=55, looking down
    # Table center at world (0.5, 0.0) -> image center
    # Red bin at world (0.72, -0.18) -> right side of image, slightly below center
    # Blue bin at world (0.72, 0.18) -> right side of image, slightly above center

    # Table region: center of image (where balls start)
    t_r0, t_r1 = int(H * 0.25), int(H * 0.75)
    t_c0, t_c1 = int(W * 0.25), int(W * 0.75)
    table_pixels = mask[t_r0:t_r1, t_c0:t_c1].sum()

    # Bin regions: right side of image
    if color == "red":
        # Red bin: world y=-0.18 -> lower half of image
        b_r0, b_r1 = int(H * 0.55), int(H * 0.85)
        b_c0, b_c1 = int(W * 0.65), int(W * 0.95)
    else:
        # Blue bin: world y=+0.18 -> upper half of image
        b_r0, b_r1 = int(H * 0.15), int(H * 0.45)
        b_c0, b_c1 = int(W * 0.65), int(W * 0.95)

    bin_pixels = mask[b_r0:b_r1, b_c0:b_c1].sum()

    return (table_pixels < table_thresh) and (bin_pixels > bin_thresh)


#  Episode runner 

def run_episode(env, renderer, vlm, agent, reprompt=False):
    """
    Run one episode. Returns trajectory info and success.
    If reprompt=True, uses vision-based completion detector to switch task string.
    """
    obs, info = env.reset()
    colors = info["ball_colors"]
    n_balls = env.n_balls

    task = generate_task_string(colors, n_balls)
    overhead, wrist = renderer.render()
    proprio = obs[0:7].astype(np.float32)
    feat = vlm.extract_features(overhead, wrist, proprio, task)

    first_sorted = False
    n_sorted = 0

    for t in range(env.max_episode_steps):
        action = agent.select_action(feat, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)

        overhead, wrist = renderer.render()
        proprio = obs[0:7].astype(np.float32)

        # Vision-based re-prompting (2-ball only)
        if reprompt and n_balls == 2 and not first_sorted:
            sorted_color = colors[0]
            if detect_ball_sorted_from_image(overhead, sorted_color):
                first_sorted = True
                remaining_color = colors[1]
                task = (f"pick up the {remaining_color} ball and "
                        f"place it in the {remaining_color} bin")

        feat = vlm.extract_features(overhead, wrist, proprio, task)

        if done or trunc:
            break

    n_sorted = info.get("n_sorted", 0)
    return n_sorted, t + 1, colors


#  Video recording 

def save_video(env, renderer_video, vlm, agent, video_path,
               camera, n_episodes, fps, reprompt=False):
    """Record episodes from a given camera and save as MP4."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (960, 720))

    successes = 0
    for ep in range(n_episodes):
        obs, info = env.reset()
        colors = info["ball_colors"]
        n_balls = env.n_balls
        task = generate_task_string(colors, n_balls)

        overhead_np, wrist_np = renderer_video.vlm_renderer.render()
        proprio = obs[0:7].astype(np.float32)
        feat = vlm.extract_features(overhead_np, wrist_np, proprio, task)

        first_sorted = False
        print(f"    Ep {ep+1}/{n_episodes}: {colors}", end="")

        for step in range(env.max_episode_steps):
            action = agent.select_action(feat, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)

            # Render video frame
            if isinstance(camera, str):
                renderer_video.video_renderer.update_scene(
                    env.data, camera=camera)
            else:
                renderer_video.video_renderer.update_scene(
                    env.data, camera=camera)
            frame = renderer_video.video_renderer.render()
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # Get VLM features
            overhead_np, wrist_np = renderer_video.vlm_renderer.render()
            proprio = obs[0:7].astype(np.float32)

            # Vision-based re-prompting
            if reprompt and n_balls == 2 and not first_sorted:
                if detect_ball_sorted_from_image(overhead_np, colors[0]):
                    first_sorted = True
                    remaining_color = colors[1]
                    task = (f"pick up the {remaining_color} ball and "
                            f"place it in the {remaining_color} bin")

            feat = vlm.extract_features(overhead_np, wrist_np, proprio, task)

            if done or trunc:
                # Hold last frame for 1.5s
                for _ in range(int(fps * 1.5)):
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                break

        n_sorted = info.get("n_sorted", 0)
        ok = n_sorted >= n_balls
        successes += int(ok)
        print(f" -> {'OK' if ok else 'NOPE'} sorted={n_sorted}/{n_balls} steps={step+1}")

    writer.release()
    print(f"    Success: {successes}/{n_episodes} ({successes/n_episodes:.0%})")


class DualRenderer:
    """
    Two renderers: one for VLM feature extraction (480x640),
    one for video recording (720x960).
    """
    def __init__(self, model, data):
        self.vlm_renderer = ImageRenderer(model, data)

        model.vis.global_.offwidth = 960
        model.vis.global_.offheight = 720
        self.video_renderer = mujoco.Renderer(model, height=720, width=960)

    def close(self):
        self.vlm_renderer.close()
        self.video_renderer.close()


#  Main 

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,
                   default="runs/vlm_sac_1ball_seed108/best_model.pt")
    p.add_argument("--vlm_checkpoint", type=str,
                   default="outputs/train/smolvla_panda_sort_v2/checkpoints/005000/pretrained_model")
    p.add_argument("--n_balls", type=int, default=1)
    p.add_argument("--n_episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--reprompt", action="store_true",
                   help="Enable vision-based re-prompting (2-ball only)")
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    if args.device:
        device = args.device
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{args.n_balls}ball"
    if args.reprompt:
        suffix += "_reprompt"

    print("=" * 60)
    print(f"VLM+SAC Visualization — {args.n_balls} ball(s)")
    print("=" * 60)
    print(f"  Checkpoint:    {args.checkpoint}")
    print(f"  Re-prompt:     {args.reprompt}")
    print(f"  Output dir:    {out_dir}")

    # Load
    print("\nLoading VLM...")
    vlm = VLMFeatureExtractor(args.vlm_checkpoint, device=device)

    print("Loading SAC agent...")
    agent = VLM_SAC(feat_dim=FEAT_DIM, act_dim=4, hidden=512, device="cpu")
    agent.load(args.checkpoint)

    env = PandaSortEnv(
        n_balls=args.n_balls,
        max_episode_steps=400 if args.n_balls == 1 else 600,
        seed=args.seed,
    )

    renderer = DualRenderer(env.model, env.data)

    #  Diagnostic run 
    print(f"\n{'='*50}")
    print(f"DIAGNOSTIC — {args.n_balls} ball(s), {args.n_episodes} episodes")
    print(f"{'='*50}")

    successes = 0
    for ep in range(args.n_episodes):
        obs, info = env.reset()
        colors = info["ball_colors"]
        task = generate_task_string(colors, args.n_balls)

        overhead, wrist = renderer.vlm_renderer.render()
        proprio = obs[0:7].astype(np.float32)
        feat = vlm.extract_features(overhead, wrist, proprio, task)

        first_sorted = False

        for step in range(env.max_episode_steps):
            action = agent.select_action(feat, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)

            overhead, wrist = renderer.vlm_renderer.render()
            proprio = obs[0:7].astype(np.float32)

            if args.reprompt and args.n_balls == 2 and not first_sorted:
                if detect_ball_sorted_from_image(overhead, colors[0]):
                    first_sorted = True
                    remaining_color = colors[1]
                    task = (f"pick up the {remaining_color} ball and "
                            f"place it in the {remaining_color} bin")

            feat = vlm.extract_features(overhead, wrist, proprio, task)

            if done or trunc:
                break

        n_sorted = info.get("n_sorted", 0)
        ok = n_sorted >= args.n_balls
        successes += int(ok)
        print(f"  Ep {ep+1}: {colors} -> {'OK' if ok else '!'} "
              f"sorted={n_sorted}/{args.n_balls} steps={step+1}")

    print(f"\n  Success: {successes}/{args.n_episodes} "
          f"({successes/args.n_episodes:.0%})")

    #  Overhead video 
    overhead_path = out_dir / f"vlm_sac{suffix}_overhead.mp4"
    print(f"\nRecording overhead -> {overhead_path}")
    save_video(env, renderer, vlm, agent, overhead_path,
               "overhead_cam", args.n_episodes, args.fps, args.reprompt)

    #  Side video 
    side_cam = mujoco.MjvCamera()
    side_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    side_cam.lookat[:] = [0.55, 0.0, 0.45]
    side_cam.distance = 1.2
    side_cam.azimuth = 160
    side_cam.elevation = -25

    side_path = out_dir / f"vlm_sac{suffix}_side.mp4"
    print(f"\nRecording side -> {side_path}")
    save_video(env, renderer, vlm, agent, side_path,
               side_cam, args.n_episodes, args.fps, args.reprompt)

    #  Front video 
    front_cam = mujoco.MjvCamera()
    front_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    front_cam.lookat[:] = [0.55, 0.0, 0.45]
    front_cam.distance = 1.0
    front_cam.azimuth = 180
    front_cam.elevation = -15

    front_path = out_dir / f"vlm_sac{suffix}_front.mp4"
    print(f"\nRecording front -> {front_path}")
    save_video(env, renderer, vlm, agent, front_path,
               front_cam, args.n_episodes, args.fps, args.reprompt)

    renderer.close()
    env.close()

    print(f"\nAll videos saved to {out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()