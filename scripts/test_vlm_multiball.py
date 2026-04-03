"""
Runs 2-ball episodes with detailed per-step logging to understand
what happens after ball 1 is placed:
    - Does the gripper hover over the bin?
    - Does it return toward the table?
    - Does it reach ball 2 but fail to grasp?
    - Does re-injecting a fresh task string help?

Mode 1 (baseline):  Same task string entire episode
Mode 2 (re-prompt): After first ball sorted, switch task string
                     to target the remaining ball specifically

Usage:
    python scripts/test_vlm_multiball.py
    python scripts/test_vlm_multiball.py --n_episodes 10
"""

import argparse
import numpy as np
import torch
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from env.panda_sort_env import (
    PandaSortEnv, OBS_DIM, PER_BALL_DIM, N_MAX_BALLS,
    TABLE_Z, BALL_RADIUS, LIFT_THRESH, BIN_POSITIONS,
)
from scripts.train_vlm_sac import (
    VLM_SAC, VLMFeatureExtractor, ImageRenderer,
    generate_task_string, FEAT_DIM,
)


def get_ball_status(obs, n_balls):
    """Extract per-ball info from state obs (for LOGGING only, not policy input)."""
    ee_pos = obs[0:3]
    balls = []
    for i in range(n_balls):
        offset = 7 + i * PER_BALL_DIM
        ball_pos = obs[offset:offset + 3]
        ball_rel = obs[offset + 3:offset + 6]
        is_sorted = obs[offset + 9] > 0.5
        dist_ee = np.linalg.norm(ball_rel)
        balls.append({
            "pos": ball_pos.copy(),
            "dist_ee": dist_ee,
            "sorted": is_sorted,
            "z": ball_pos[2],
        })
    return ee_pos, balls


def run_diagnostic_episode(env, renderer, vlm, agent, ep_num,
                           ball_colors, mode="baseline"):
    """
    Run one 2-ball episode with detailed logging.

    mode="baseline":   same task string throughout
    mode="re_prompt":  switch task string after first ball sorted
    """
    obs, info = env.reset()
    n_balls = env.n_balls
    colors = info["ball_colors"]

    task = generate_task_string(colors, n_balls)
    print(f"\n  Ep {ep_num}: {colors} | mode={mode} | task=\"{task}\"")

    overhead, wrist = renderer.render()
    proprio = obs[0:7].astype(np.float32)
    feat = vlm.extract_features(overhead, wrist, proprio, task)

    first_sorted_step = None
    second_sorted_step = None
    phase = "sorting_first"
    re_prompted = False

    for t in range(env.max_episode_steps):
        action = agent.select_action(feat, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)

        # Check ball status (state obs for LOGGING only)
        ee_pos, balls = get_ball_status(obs, n_balls)
        n_sorted = sum(b["sorted"] for b in balls)

        # Detect first ball sorted
        if n_sorted >= 1 and first_sorted_step is None:
            first_sorted_step = t
            unsorted = [i for i, b in enumerate(balls) if not b["sorted"]]
            print(f"    Step {t:3d}: FIRST BALL SORTED! "
                  f"ee=[{ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}]")
            if unsorted:
                b2 = balls[unsorted[0]]
                print(f"             Remaining ball {unsorted[0]} at "
                      f"[{b2['pos'][0]:.3f},{b2['pos'][1]:.3f},{b2['pos'][2]:.3f}] "
                      f"dist_ee={b2['dist_ee']:.3f}")
            phase = "after_first"

            # Re-prompt mode: switch task string to target remaining ball
            if mode == "re_prompt" and unsorted:
                remaining_color = colors[unsorted[0]]
                task = f"pick up the {remaining_color} ball and place it in the {remaining_color} bin"
                re_prompted = True
                print(f"             RE-PROMPT -> \"{task}\"")

        # Detect second ball sorted
        if n_sorted >= 2 and second_sorted_step is None:
            second_sorted_step = t
            print(f"    Step {t:3d}: SECOND BALL SORTED!")

        # Log every 50 steps after first ball sorted
        if phase == "after_first" and (t - first_sorted_step) % 50 == 0 and t > first_sorted_step:
            unsorted = [i for i, b in enumerate(balls) if not b["sorted"]]
            if unsorted:
                b2 = balls[unsorted[0]]
                fw = obs[6]
                print(f"    Step {t:3d}: ee=[{ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}] "
                      f"ball2_dist={b2['dist_ee']:.3f} ball2_z={b2['z']:.3f} "
                      f"fw={fw:.4f} reward={reward:.2f}")

        # Re-extract features
        overhead, wrist = renderer.render()
        proprio = obs[0:7].astype(np.float32)
        feat = vlm.extract_features(overhead, wrist, proprio, task)

        if done or trunc:
            break

    n_sorted_final = info.get("n_sorted", 0)
    status = "ALL" if n_sorted_final >= n_balls else f"{n_sorted_final}/{n_balls}"
    print(f"    RESULT: {status} | first_sorted={first_sorted_step} "
          f"second_sorted={second_sorted_step} | steps={t+1}")

    return {
        "n_sorted": n_sorted_final,
        "all_sorted": n_sorted_final >= n_balls,
        "first_sorted_step": first_sorted_step,
        "second_sorted_step": second_sorted_step,
        "total_steps": t + 1,
        "re_prompted": re_prompted,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,
                   default="runs/vlm_sac_1ball_seed108/best_model.pt")
    p.add_argument("--vlm_checkpoint", type=str,
                   default="outputs/train/smolvla_panda_sort_v2/checkpoints/005000/pretrained_model",
                   help="SmolVLA checkpoint for frozen VLM features (must match training)")
    p.add_argument("--n_episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    cfg = p.parse_args()

    if cfg.device:
        device = cfg.device
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("=" * 60)
    print("VLM+SAC 2-BALL DIAGNOSTIC")
    print("=" * 60)

    vlm = VLMFeatureExtractor(cfg.vlm_checkpoint, device=device)
    agent = VLM_SAC(feat_dim=FEAT_DIM, act_dim=4, hidden=512, device="cpu")
    agent.load(cfg.checkpoint)

    env = PandaSortEnv(
        n_balls=2,
        reward_mode="dense",
        max_episode_steps=400,
        seed=cfg.seed,
    )
    renderer = ImageRenderer(env.model, env.data)

    #  Mode 1: Baseline (same task string) 
    print(f"\n{'='*50}")
    print("MODE 1: BASELINE (same task string entire episode)")
    print(f"{'='*50}")

    baseline_results = []
    for ep in range(cfg.n_episodes):
        r = run_diagnostic_episode(
            env, renderer, vlm, agent, ep + 1,
            ball_colors=None, mode="baseline")
        baseline_results.append(r)

    #  Mode 2: Re-prompt after first ball 
    print(f"\n{'='*50}")
    print("MODE 2: RE-PROMPT (switch task string after first ball sorted)")
    print(f"{'='*50}")

    # Reset env seed for fair comparison
    env.close()
    env = PandaSortEnv(
        n_balls=2,
        reward_mode="dense",
        max_episode_steps=400,
        seed=cfg.seed,
    )
    renderer = ImageRenderer(env.model, env.data)

    reprompt_results = []
    for ep in range(cfg.n_episodes):
        r = run_diagnostic_episode(
            env, renderer, vlm, agent, ep + 1,
            ball_colors=None, mode="re_prompt")
        reprompt_results.append(r)

    #  Summary 
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for name, results in [("Baseline", baseline_results),
                          ("Re-prompt", reprompt_results)]:
        n = len(results)
        all_sorted = sum(r["all_sorted"] for r in results)
        got_first = sum(r["first_sorted_step"] is not None for r in results)
        got_second = sum(r["second_sorted_step"] is not None for r in results)

        first_steps = [r["first_sorted_step"] for r in results
                       if r["first_sorted_step"] is not None]
        avg_first = np.mean(first_steps) if first_steps else float('nan')

        print(f"\n  {name}:")
        print(f"    All sorted:      {all_sorted}/{n} ({all_sorted/n:.0%})")
        print(f"    Got first ball:  {got_first}/{n} ({got_first/n:.0%})")
        print(f"    Got second ball: {got_second}/{n} ({got_second/n:.0%})")
        print(f"    Avg steps to 1st: {avg_first:.0f}")

    if reprompt_results:
        b_rate = sum(r["all_sorted"] for r in baseline_results) / len(baseline_results)
        r_rate = sum(r["all_sorted"] for r in reprompt_results) / len(reprompt_results)
        print(f"\n  VERDICT:")
        if r_rate > b_rate + 0.15:
            print(f"  Re-prompting helps significantly ({b_rate:.0%} -> {r_rate:.0%})")
            print(f"  -> High-level just needs to swap task strings")
        elif r_rate > b_rate:
            print(f"  Re-prompting helps slightly ({b_rate:.0%} -> {r_rate:.0%})")
            print(f"  -> Language helps but isn't the only bottleneck")
        else:
            print(f"  Re-prompting doesn't help ({b_rate:.0%} -> {r_rate:.0%})")
            print(f"  -> Problem is behavioral (policy stalls), not language")

    renderer.close()
    env.close()


if __name__ == "__main__":
    main()