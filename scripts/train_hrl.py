"""
Train hierarchical RL for multi-ball sorting.

High-level DQN learns which ball to target.
Low-level SAC (frozen 1-ball checkpoint) executes pick-and-place.

Usage:
    python scripts/train_hrl.py --low_level runs/sac_1ball_2/best_model.pt --n_balls 2
    python scripts/train_hrl.py --low_level runs/sac_1ball_2/best_model.pt --n_balls 4
    python scripts/train_hrl.py --test --checkpoint runs/hrl_2ball/best_model.pt
"""

import argparse
import numpy as np
import torch
import time
import csv
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from env.panda_sort_env import (
    PandaSortEnv, OBS_DIM, PER_BALL_DIM, N_MAX_BALLS
)
from scripts.train_sac import SAC
from hrl_controller import (
    HierarchicalController, extract_hl_obs, get_valid_mask
)


#  Evaluation 

def evaluate(env, hrl, n_episodes=20):
    """Evaluate the full hierarchical system."""
    results = {
        "all_sorted": 0,
        "n_sorted_total": 0,
        "n_balls_total": 0,
        "avg_steps": 0,
        "avg_reward": 0,
        "hl_decisions": 0,
    }

    for ep in range(n_episodes):
        obs, info = env.reset()
        hrl.reset()
        ep_reward = 0
        hl_calls = 0

        for t in range(env.max_episode_steps):
            # High-level: pick target if needed
            if hrl.needs_new_target(obs):
                valid = get_valid_mask(obs)
                if not np.any(valid):
                    break  # all sorted
                hrl.select_target(obs, epsilon=0.0)  # greedy
                hl_calls += 1

            # Low-level: execute
            action = hrl.select_action(obs)
            obs, reward, done, trunc, info = env.step(action)
            ep_reward += reward

            if done or trunc:
                break

        n_sorted = info.get("n_sorted", 0)
        results["all_sorted"] += int(n_sorted >= env.n_balls)
        results["n_sorted_total"] += n_sorted
        results["n_balls_total"] += env.n_balls
        results["avg_steps"] += t + 1
        results["avg_reward"] += ep_reward
        results["hl_decisions"] += hl_calls

    n = n_episodes
    return {
        "success_rate": results["all_sorted"] / n,
        "sort_rate": results["n_sorted_total"] / results["n_balls_total"],
        "avg_steps": results["avg_steps"] / n,
        "avg_reward": results["avg_reward"] / n,
        "avg_hl_decisions": results["hl_decisions"] / n,
    }


#  Heuristic baseline 

def evaluate_heuristic(env, low_level, n_episodes=20, max_steps_per_ball=150):
    """
    Baseline: always pick closest unsorted ball (the test_2ball logic).
    This is the bar the learned HL policy needs to beat.
    """
    results = {"all_sorted": 0, "n_sorted_total": 0, "n_balls_total": 0}

    for ep in range(n_episodes):
        obs, info = env.reset()
        current_target = None
        steps_on_target = 0

        for t in range(env.max_episode_steps):
            # Pick closest unsorted
            need_new = (current_target is None or
                        steps_on_target >= max_steps_per_ball)
            if current_target is not None:
                offset = 7 + current_target * PER_BALL_DIM
                if obs[offset + 9] > 0.5:  # sorted
                    need_new = True

            if need_new:
                ee_pos = obs[0:3]
                best_idx, best_dist = None, np.inf
                for i in range(env.n_balls):
                    offset = 7 + i * PER_BALL_DIM
                    if obs[offset + 9] > 0.5 or obs[offset + 10] < 0.5:
                        continue
                    d = np.linalg.norm(obs[offset + 3:offset + 6])
                    if d < best_dist:
                        best_dist = d
                        best_idx = i
                if best_idx is None:
                    break
                current_target = best_idx
                steps_on_target = 0

            # Remap and execute
            from hrl_controller import make_1ball_obs
            fake_obs = make_1ball_obs(obs, current_target)
            action = low_level.select_action(fake_obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)
            steps_on_target += 1

            if done or trunc:
                break

        n_sorted = info.get("n_sorted", 0)
        results["all_sorted"] += int(n_sorted >= env.n_balls)
        results["n_sorted_total"] += n_sorted
        results["n_balls_total"] += env.n_balls

    n = n_episodes
    return {
        "success_rate": results["all_sorted"] / n,
        "sort_rate": results["n_sorted_total"] / results["n_balls_total"],
    }


#  Training loop 

def train(cfg):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Run dir
    name = cfg.run_name or f"hrl_{cfg.n_balls}ball_seed{cfg.seed}"
    run_dir = Path("runs") / name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(cfg), f, indent=2)

    # Env
    env = PandaSortEnv(
        n_balls=cfg.n_balls,
        reward_mode="dense",
        max_episode_steps=cfg.n_balls * 150 + 50,  # scale with balls
        seed=cfg.seed,
    )

    # Low-level (frozen)
    low_level = SAC(obs_dim=OBS_DIM, act_dim=4, hidden=256, device="cpu")
    low_level.load(cfg.low_level)
    for p in low_level.actor.parameters():
        p.requires_grad = False
    print(f"Loaded frozen low-level from {cfg.low_level}")

    # High-level
    hrl = HierarchicalController(
        low_level, n_balls=cfg.n_balls,
        max_steps_per_ball=cfg.max_steps_per_ball,
    )

    # Heuristic baseline
    print("\nRunning heuristic baseline (closest-ball)...")
    baseline = evaluate_heuristic(env, low_level, n_episodes=50,
                                  max_steps_per_ball=cfg.max_steps_per_ball)
    print(f"  Heuristic: success={baseline['success_rate']:.0%}, "
          f"sort_rate={baseline['sort_rate']:.0%}")
    print(f"  This is the bar to beat.\n")

    # CSV logger
    csv_path = run_dir / "eval_log.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_fields = ["episode", "success_rate", "sort_rate", "avg_steps",
                  "avg_reward", "avg_hl_decisions", "epsilon", "hl_loss"]
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    print("=" * 60)
    print(f"HRL Training — {cfg.n_balls} balls")
    print(f"=" * 60)
    print(f"  Run dir:          {run_dir}")
    print(f"  Episodes:         {cfg.n_episodes}")
    print(f"  Max steps/ball:   {cfg.max_steps_per_ball}")
    print(f"  Epsilon:          {cfg.eps_start} -> {cfg.eps_end}")
    print(f"  HL update every:  {cfg.hl_update_every} HL transitions")

    best_success = 0.0
    t_start = time.time()

    for episode in range(1, cfg.n_episodes + 1):
        # Epsilon schedule
        frac = min(1.0, episode / (cfg.n_episodes * 0.7))
        epsilon = cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)

        obs, info = env.reset()
        hrl.reset()
        ep_reward = 0
        hl_transitions = []  # collect (hl_obs, action, ...) for this episode

        # Track high-level state for reward assignment
        prev_n_sorted = 0
        prev_hl_obs = None
        prev_hl_action = None
        prev_valid_mask = None

        for t in range(env.max_episode_steps):
            # High-level decision
            if hrl.needs_new_target(obs):
                valid = get_valid_mask(obs)
                if not np.any(valid):
                    # All sorted — store terminal transition for previous HL
                    if prev_hl_obs is not None:
                        n_now = info.get("n_sorted", 0)
                        hl_reward = float(n_now - prev_n_sorted)
                        if n_now >= env.n_balls:
                            hl_reward += 5.0  # all sorted bonus
                        hl_obs_now = extract_hl_obs(obs, env.n_balls)
                        hrl.hl_buffer.add(
                            prev_hl_obs, prev_hl_action, hl_reward,
                            hl_obs_now, True, valid
                        )
                    break

                # Store transition for PREVIOUS high-level decision
                if prev_hl_obs is not None:
                    n_now = info.get("n_sorted", 0)
                    hl_reward = float(n_now - prev_n_sorted)
                    hl_obs_now = extract_hl_obs(obs, env.n_balls)
                    hrl.hl_buffer.add(
                        prev_hl_obs, prev_hl_action, hl_reward,
                        hl_obs_now, False, valid
                    )
                    prev_n_sorted = n_now

                # New high-level decision
                hl_obs = extract_hl_obs(obs, env.n_balls)
                target = hrl.select_target(obs, epsilon=epsilon)
                prev_hl_obs = hl_obs
                prev_hl_action = target
                prev_valid_mask = valid

            # Low-level action
            action = hrl.select_action(obs)
            obs, reward, done, trunc, info = env.step(action)
            ep_reward += reward

            if done or trunc:
                # Store final HL transition
                if prev_hl_obs is not None:
                    n_now = info.get("n_sorted", 0)
                    hl_reward = float(n_now - prev_n_sorted)
                    if n_now >= env.n_balls:
                        hl_reward += 5.0
                    hl_obs_now = extract_hl_obs(obs, env.n_balls)
                    valid = get_valid_mask(obs)
                    hrl.hl_buffer.add(
                        prev_hl_obs, prev_hl_action, hl_reward,
                        hl_obs_now, True, valid
                    )
                break

        # Update high-level DQN
        hl_loss = 0.0
        if hrl.hl_buffer.size >= cfg.hl_batch_size:
            for _ in range(cfg.hl_updates_per_ep):
                logs = hrl.update_hl(cfg.hl_batch_size)
                hl_loss = logs.get("hl_loss", 0.0)

        # Eval
        if episode % cfg.eval_every == 0:
            rates = evaluate(env, hrl, n_episodes=cfg.n_eval_eps)

            print(f"\n  {'='*50}")
            print(f"  EVAL @ Episode {episode}")
            print(f"  {'='*50}")
            print(f"    All sorted:    {rates['success_rate']:.0%}")
            print(f"    Per-ball sort:  {rates['sort_rate']:.0%}")
            print(f"    Avg steps:      {rates['avg_steps']:.0f}")
            print(f"    Avg reward:     {rates['avg_reward']:.1f}")
            print(f"    HL decisions:   {rates['avg_hl_decisions']:.1f}")
            print(f"    Epsilon:        {epsilon:.3f}")
            print(f"    Heuristic bar:  {baseline['success_rate']:.0%}")
            print(f"  {'='*50}\n")

            row = {
                "episode": episode,
                "epsilon": f"{epsilon:.3f}",
                "hl_loss": f"{hl_loss:.4f}",
                **{k: f"{v:.4f}" for k, v in rates.items()},
            }
            csv_writer.writerow(row)
            csv_file.flush()

            if rates["success_rate"] > best_success:
                best_success = rates["success_rate"]
                hrl.save(str(run_dir / "best_model.pt"))
                print(f"    New best! {best_success:.0%}")

        # Progress
        if episode % 50 == 0:
            elapsed = time.time() - t_start
            eps_per_sec = episode / elapsed
            print(f"  Ep {episode:5d} | "
                  f"Reward {ep_reward:7.1f} | "
                  f"Sorted {info.get('n_sorted', 0)}/{env.n_balls} | "
                  f"Eps {epsilon:.3f} | "
                  f"Buffer {hrl.hl_buffer.size} | "
                  f"EPS {eps_per_sec:.1f}")

    # Final
    hrl.save(str(run_dir / "final_model.pt"))
    csv_file.close()

    print(f"\n{'='*60}")
    print("FINAL EVALUATION (50 episodes)")
    print(f"{'='*60}")
    final = evaluate(env, hrl, n_episodes=50)
    print(f"  All sorted:    {final['success_rate']:.0%}")
    print(f"  Per-ball sort:  {final['sort_rate']:.0%}")
    print(f"  Avg steps:      {final['avg_steps']:.0f}")
    print(f"  Heuristic:      {baseline['success_rate']:.0%}")

    elapsed = time.time() - t_start
    print(f"\n  Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Best: {best_success:.0%}")

    env.close()


#  Test mode 

def test(cfg):
    env = PandaSortEnv(
        n_balls=cfg.n_balls,
        reward_mode="dense",
        max_episode_steps=cfg.n_balls * 150 + 50,
        seed=cfg.seed,
    )

    low_level = SAC(obs_dim=OBS_DIM, act_dim=4, hidden=256, device="cpu")
    low_level.load(cfg.low_level)

    hrl = HierarchicalController(
        low_level, n_balls=cfg.n_balls,
        max_steps_per_ball=cfg.max_steps_per_ball,
    )
    hrl.load(cfg.checkpoint)

    print("=" * 60)
    print(f"HRL TEST — {cfg.n_balls} balls")
    print("=" * 60)

    rates = evaluate(env, hrl, n_episodes=50)
    print(f"  All sorted:    {rates['success_rate']:.0%}")
    print(f"  Per-ball sort:  {rates['sort_rate']:.0%}")
    print(f"  Avg steps:      {rates['avg_steps']:.0f}")

    # Also run heuristic for comparison
    baseline = evaluate_heuristic(env, low_level, n_episodes=50,
                                  max_steps_per_ball=cfg.max_steps_per_ball)
    print(f"  Heuristic:      {baseline['success_rate']:.0%}")

    env.close()


#  Config 

def get_config():
    p = argparse.ArgumentParser()
    p.add_argument("--low_level", type=str, required=True,
                   help="Path to frozen 1-ball SAC checkpoint")
    p.add_argument("--n_balls", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_name", type=str, default=None)

    # Training
    p.add_argument("--n_episodes", type=int, default=2000)
    p.add_argument("--max_steps_per_ball", type=int, default=150)
    p.add_argument("--eps_start", type=float, default=1.0)
    p.add_argument("--eps_end", type=float, default=0.05)

    # DQN
    p.add_argument("--hl_batch_size", type=int, default=64)
    p.add_argument("--hl_update_every", type=int, default=1)
    p.add_argument("--hl_updates_per_ep", type=int, default=4)

    # Eval
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--n_eval_eps", type=int, default=20)

    # Test mode
    p.add_argument("--test", action="store_true")
    p.add_argument("--checkpoint", type=str, default=None)

    return p.parse_args()


if __name__ == "__main__":
    cfg = get_config()
    if cfg.test:
        assert cfg.checkpoint, "Need --checkpoint for test mode"
        test(cfg)
    else:
        train(cfg)