"""
Diagnostic eval for SmolVLA — prints everything to find the failure mode.

Run this, paste the output back. It will:
    1. Print raw action outputs (pre and post processing)
    2. Print action normalization stats from the dataset
    3. Save sample images to disk (what the policy actually sees)
    4. Check action chunking behavior
    5. Run only 2 episodes with verbose output

Usage:
    python scripts/eval_smolvla_diagnostic.py
    python scripts/eval_smolvla_diagnostic.py --checkpoint <path>
"""

import argparse
import numpy as np
import torch
import mujoco
import time
from pathlib import Path
import sys
import PIL.Image
import json

sys.path.append(str(Path(__file__).parent.parent))
from env.panda_sort_env import (
    PandaSortEnv, OBS_DIM, PER_BALL_DIM, N_MAX_BALLS,
    TABLE_Z, BALL_RADIUS, BIN_POSITIONS, POS_SCALE,
)

IMG_H, IMG_W = 480, 640
DIAG_DIR = Path("diag_smolvla")


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,
                   default="outputs/train/smolvla_panda_sort_v2/checkpoints/005000/pretrained_model")
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--seed", type=int, default=42)
    cfg = p.parse_args()

    DIAG_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("SmolVLA DIAGNOSTIC EVAL")
    print("=" * 70)
    print(f"  Checkpoint: {cfg.checkpoint}")
    print(f"  Device:     {cfg.device}")

    #  1. Check checkpoint contents 
    ckpt_path = Path(cfg.checkpoint)
    print(f"\n{'='*70}")
    print("STEP 1: Checkpoint structure")
    print(f"{'='*70}")
    print(f"  Path exists: {ckpt_path.exists()}")
    if ckpt_path.exists():
        contents = list(ckpt_path.iterdir()) if ckpt_path.is_dir() else [ckpt_path]
        for f in sorted(contents)[:20]:
            size = f.stat().st_size if f.is_file() else 0
            print(f"    {f.name:40s} {size/1024:.1f} KB")

    #  2. Check action stats 
    print(f"\n{'='*70}")
    print("STEP 2: Action normalization stats")
    print(f"{'='*70}")

    # Look for stats in multiple possible locations
    stats_locations = [
        ckpt_path / "stats.json",
        ckpt_path / "meta" / "stats.json",
        ckpt_path.parent / "stats.json",
        ckpt_path.parent / "meta" / "stats.json",
    ]
    # Also check the training output dir
    train_dir = Path("outputs/train/smolvla_panda_sort")
    if train_dir.exists():
        stats_locations.extend([
            train_dir / "stats.json",
            train_dir / "meta" / "stats.json",
        ])
    # Check the dataset itself
    dataset_dirs = [
        Path("panda_sort_merged/panda_sort_all"),
        Path("1_ball_demos/demos"),
    ]
    for dd in dataset_dirs:
        stats_locations.extend([
            dd / "meta" / "stats.json",
            dd / "stats.json",
        ])

    found_stats = False
    for sp in stats_locations:
        if sp.exists():
            print(f"  Found stats at: {sp}")
            with open(sp) as f:
                stats = json.load(f)
            # Print action stats
            for key in sorted(stats.keys()):
                if "action" in key.lower():
                    print(f"    {key}:")
                    val = stats[key]
                    if isinstance(val, dict):
                        for k2, v2 in val.items():
                            if isinstance(v2, list):
                                print(f"      {k2}: {[f'{x:.4f}' for x in v2]}")
                            else:
                                print(f"      {k2}: {v2}")
                    else:
                        print(f"      {val}")
            # Also print state stats
            for key in sorted(stats.keys()):
                if "state" in key.lower():
                    print(f"    {key}:")
                    val = stats[key]
                    if isinstance(val, dict):
                        for k2, v2 in val.items():
                            if isinstance(v2, list):
                                print(f"      {k2}: {[f'{x:.4f}' for x in v2]}")
                            else:
                                print(f"      {k2}: {v2}")
            found_stats = True
            break

    if not found_stats:
        print("  WARNING: No stats.json found in standard locations")
        # Stats are baked into the pre/post processor safetensors
        # Let's read the processor JSON configs directly
        for name in ["policy_preprocessor.json", "policy_postprocessor.json"]:
            pp_path = ckpt_path / name
            if pp_path.exists():
                print(f"\n  {name}:")
                with open(pp_path) as f:
                    pp_config = json.load(f)
                print(f"    {json.dumps(pp_config, indent=2)[:2000]}")

        # Also try loading the safetensors to see the actual normalization values
        unnorm_path = ckpt_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        if unnorm_path.exists():
            from safetensors.torch import load_file
            unnorm_data = load_file(str(unnorm_path))
            print(f"\n  Postprocessor (unnormalizer) tensors:")
            for k, v in sorted(unnorm_data.items()):
                print(f"    {k}: shape={v.shape}, values={v.numpy()}")

        norm_path = ckpt_path / "policy_preprocessor_step_5_normalizer_processor.safetensors"
        if norm_path.exists():
            from safetensors.torch import load_file
            norm_data = load_file(str(norm_path))
            print(f"\n  Preprocessor (normalizer) tensors:")
            for k, v in sorted(norm_data.items()):
                print(f"    {k}: shape={v.shape}, values={v.numpy()}")

    #  3. Load model 
    print(f"\n{'='*70}")
    print("STEP 3: Load SmolVLA")
    print(f"{'='*70}")

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    device = torch.device(cfg.device)
    policy = SmolVLAPolicy.from_pretrained(cfg.checkpoint).to(device).eval()

    print(f"  Model loaded successfully")
    print(f"  chunk_size:       {policy.config.chunk_size}")
    print(f"  n_action_steps:   {policy.config.n_action_steps}")

    # Dump ALL config keys — we need to see what SmolVLA actually expects
    config_dict = policy.config.to_dict() if hasattr(policy.config, 'to_dict') else vars(policy.config)
    print(f"\n  ALL config keys:")
    for k in sorted(config_dict.keys()):
        v = config_dict[k]
        v_str = repr(v)
        if len(v_str) > 120:
            v_str = v_str[:120] + "..."
        print(f"    {k}: {v_str}")

    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        cfg.checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    #  4. Run diagnostic episodes 
    print(f"\n{'='*70}")
    print("STEP 4: Run 2 diagnostic episodes")
    print(f"{'='*70}")

    env = PandaSortEnv(
        n_balls=1,
        reward_mode="dense",
        max_episode_steps=200,
        seed=cfg.seed,
    )
    renderer = EvalRenderer(env.model, env.data)

    for ep in range(2):
        obs, info = env.reset()
        ball_colors = info["ball_colors"]
        task = f"pick up the {ball_colors[0]} ball and place it in the {ball_colors[0]} bin"

        print(f"\n  --- Episode {ep+1} ---")
        print(f"  Ball colors: {ball_colors}")
        print(f"  Task: \"{task}\"")

        action_buffer = []
        action_idx = 0

        for t in range(min(20, env.max_episode_steps)):  # Only 20 steps for diagnostic
            overhead_img, wrist_img = renderer.render()
            state = obs[0:6].astype(np.float32)

            # Save images on first step
            if t == 0 and ep == 0:
                overhead_img.save(DIAG_DIR / "overhead_step0.png")
                wrist_img.save(DIAG_DIR / "wrist_step0.png")
                print(f"  Saved sample images to {DIAG_DIR}/")
                print(f"  Overhead size: {overhead_img.size}")
                print(f"  Wrist size:    {wrist_img.size}")

            # Check if we need new predictions
            if action_idx >= len(action_buffer):
                # Build batch
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
                    "task": task,
                }

                if t < 3:
                    print(f"\n  Step {t} — PRE-PREPROCESS batch:")
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            print(f"    {k}: shape={v.shape}, dtype={v.dtype}, "
                                  f"range=[{v.min():.4f}, {v.max():.4f}]")
                        else:
                            print(f"    {k}: {repr(v)[:80]}")

                # Preprocess
                batch_pp = preprocess(batch)

                if t < 3:
                    print(f"\n  Step {t} — POST-PREPROCESS batch:")
                    for k, v in batch_pp.items():
                        if isinstance(v, torch.Tensor):
                            print(f"    {k}: shape={v.shape}, dtype={v.dtype}, "
                                  f"range=[{v.min().item():.4f}, {v.max().item():.4f}], "
                                  f"device={v.device}")
                        else:
                            print(f"    {k}: {repr(v)[:80]}")

                # Forward pass
                with torch.no_grad():
                    raw_output = policy.select_action(batch_pp)

                if t < 3:
                    print(f"\n  Step {t} — RAW MODEL OUTPUT:")
                    if isinstance(raw_output, dict):
                        for k, v in raw_output.items():
                            if isinstance(v, torch.Tensor):
                                print(f"    {k}: shape={v.shape}, "
                                      f"range=[{v.min().item():.4f}, {v.max().item():.4f}]")
                                if v.numel() <= 30:
                                    print(f"    {k} values: {v.cpu().numpy()}")
                            else:
                                print(f"    {k}: {v}")
                    elif isinstance(raw_output, torch.Tensor):
                        print(f"    tensor: shape={raw_output.shape}, "
                              f"range=[{raw_output.min().item():.4f}, {raw_output.max().item():.4f}]")
                        if raw_output.numel() <= 60:
                            print(f"    values:\n{raw_output.cpu().numpy()}")
                    else:
                        print(f"    type={type(raw_output)}, value={raw_output}")

                # Postprocess
                post_output = postprocess(raw_output)

                if t < 3:
                    print(f"\n  Step {t} — POST-POSTPROCESS output:")
                    if isinstance(post_output, dict):
                        for k, v in post_output.items():
                            if isinstance(v, torch.Tensor):
                                print(f"    {k}: shape={v.shape}, "
                                      f"range=[{v.min().item():.4f}, {v.max().item():.4f}]")
                                if v.numel() <= 30:
                                    print(f"    {k} values: {v.cpu().numpy()}")
                    elif isinstance(post_output, torch.Tensor):
                        print(f"    tensor: shape={post_output.shape}, "
                              f"range=[{post_output.min().item():.4f}, {post_output.max().item():.4f}]")
                        if post_output.numel() <= 60:
                            print(f"    values:\n{post_output.cpu().numpy()}")

                # Extract actions
                if isinstance(post_output, dict):
                    actions = post_output["action"].cpu().numpy()
                elif isinstance(post_output, torch.Tensor):
                    actions = post_output.cpu().numpy()
                else:
                    actions = np.array(post_output)
                if actions.ndim == 1:
                    actions = actions[np.newaxis, :]

                n_action_steps = min(policy.config.n_action_steps, len(actions))
                action_buffer = [actions[i] for i in range(n_action_steps)]
                action_idx = 0

                if t < 3:
                    print(f"\n  Action buffer size: {len(action_buffer)}")
                    print(f"  First 3 actions in chunk:")
                    for ai in range(min(3, len(action_buffer))):
                        print(f"    [{ai}]: {action_buffer[ai]}")

            # Get current action
            action_6d = action_buffer[action_idx]
            action_idx += 1

            # Extract 4D
            action_4d = action_6d[:4] if len(action_6d) >= 4 else np.zeros(4)
            action_4d_clipped = np.clip(action_4d, -1.0, 1.0)

            if t < 10:
                ee_pos = obs[0:3]
                ball_pos = obs[7:10]
                print(f"  t={t:3d}: action_4d={np.array2string(action_4d, precision=3, suppress_small=True):40s} "
                      f"clipped={np.array2string(action_4d_clipped, precision=3, suppress_small=True):40s} "
                      f"ee={np.array2string(ee_pos, precision=3):25s} "
                      f"ball={np.array2string(ball_pos, precision=3)}")

            obs, reward, done, trunc, info = env.step(action_4d_clipped)

        print(f"  Final: sorted={info.get('n_sorted', 0)}/1")

    #  5. Compare with oracle 
    print(f"\n{'='*70}")
    print("STEP 5: Oracle sanity check (SAC, 5 episodes)")
    print(f"{'='*70}")

    sys.path.append(str(Path(__file__).parent))
    from train_sac import SAC as SACAgent

    oracle = SACAgent(obs_dim=OBS_DIM, act_dim=4, hidden=256, device="cpu")
    oracle_path = "runs/sac_1ball_new/final_model.pt"
    if Path(oracle_path).exists():
        oracle.load(oracle_path)
        for oep in range(5):
            obs, info = env.reset()
            for t in range(200):
                action = oracle.select_action(obs, deterministic=True)
                obs, r, done, trunc, info = env.step(action)
                if done or trunc:
                    break
            print(f"  Oracle ep {oep+1}: sorted={info.get('n_sorted', 0)}/1, steps={t+1}")
    else:
        print(f"  Oracle checkpoint not found at {oracle_path}")

    renderer.close()
    env.close()

    print(f"\n{'='*70}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'='*70}")
    print(f"  Check {DIAG_DIR}/ for saved images")
    print(f"  Paste this entire output back to Claude")


if __name__ == "__main__":
    main()