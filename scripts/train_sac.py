"""
Full SAC training on PandaSortEnv.

Usage:
    python scripts/train_sac.py
    python scripts/train_sac.py --total_steps 500000
    python scripts/train_sac.py --resume runs/sac_v2/checkpoint_100000.pt
    python scripts/train_sac.py --n_balls 2
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from collections import deque
from pathlib import Path
import time
import csv
import json
import os
import sys

sys.path.append(str(Path(__file__).parent.parent))
from env.panda_sort_env import PandaSortEnv, TABLE_Z, OBS_DIM, PER_BALL_DIM, N_MAX_BALLS, BALL_RADIUS, LIFT_THRESH


#  Replay Buffer 

class ReplayBuffer:
    def __init__(self, capacity=200_000, obs_dim=25, act_dim=4):
        self.capacity = capacity
        self.obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action   = np.zeros((capacity, act_dim), dtype=np.float32)
        self.reward   = np.zeros((capacity, 1),       dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done     = np.zeros((capacity, 1),       dtype=np.float32)
        self._ptr  = 0
        self._size = 0

    def add(self, obs, action, reward, next_obs, done):
        i = self._ptr
        self.obs[i]      = obs
        self.action[i]   = action
        self.reward[i]   = reward
        self.next_obs[i] = next_obs
        self.done[i]     = float(done)
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size, device="cpu"):
        idx = np.random.randint(0, self._size, batch_size)
        return (
            torch.from_numpy(self.obs[idx]).to(device),
            torch.from_numpy(self.action[idx]).to(device),
            torch.from_numpy(self.reward[idx]).to(device),
            torch.from_numpy(self.next_obs[idx]).to(device),
            torch.from_numpy(self.done[idx]).to(device),
        )

    @property
    def size(self):
        return self._size


#  Networks 

class Actor(nn.Module):
    def __init__(self, obs_dim=25, act_dim=4, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.LayerNorm(hidden), nn.ReLU(),
        )
        self.mean    = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
        self._init_weights()

    def forward(self, obs):
        h = self.net(obs)
        return self.mean(h), self.log_std(h).clamp(-5, 2)

    def sample(self, obs):
        mean, log_std = self(obs)
        std = log_std.exp()
        x = mean + torch.randn_like(mean) * std
        action = torch.tanh(x)
        log_prob = (
            -0.5 * ((x - mean) / (std + 1e-8)).pow(2)
            - log_std - 0.5 * np.log(2 * np.pi)
        ).sum(-1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(-1, keepdim=True)
        return action, log_prob

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)


class Critic(nn.Module):
    def __init__(self, obs_dim=25, act_dim=4, hidden=256):
        super().__init__()
        dim = obs_dim + act_dim

        def make_q():
            return nn.Sequential(
                nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        self.q1 = make_q()
        self.q2 = make_q()

    def forward(self, obs, action):
        x = torch.cat([obs, action], -1)
        return self.q1(x), self.q2(x)


#  SAC Agent 

class SAC:
    def __init__(self, obs_dim=25, act_dim=4, hidden=256,
                 lr=3e-4, gamma=0.99, tau=0.005, alpha=0.1,
                 device="cpu"):
        self.device = torch.device(device)
        self.gamma  = gamma
        self.tau    = tau
        self.act_dim = act_dim

        self.actor  = Actor(obs_dim, act_dim, hidden).to(self.device)
        self.critic = Critic(obs_dim, act_dim, hidden).to(self.device)
        self.critic_target = deepcopy(self.critic).to(self.device)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.log_alpha = torch.tensor(
            np.log(alpha), requires_grad=True,
            dtype=torch.float32, device=self.device)
        self.target_entropy = -float(act_dim)

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.alpha_opt  = torch.optim.Adam([self.log_alpha], lr=lr)

        self.total_updates = 0

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    @torch.no_grad()
    def select_action(self, obs, deterministic=False):
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        if deterministic:
            mean, _ = self.actor(obs_t)
            action = torch.tanh(mean)
        else:
            action, _ = self.actor.sample(obs_t)
        return action.squeeze(0).cpu().numpy()

    def update(self, buffer, batch_size=256):
        self.total_updates += 1
        obs, action, reward, next_obs, done = buffer.sample(batch_size, self.device)

        # Critic
        with torch.no_grad():
            next_action, next_lp = self.actor.sample(next_obs)
            q1_next, q2_next = self.critic_target(next_obs, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_lp
            q_target = reward + self.gamma * (1 - done) * q_next

        q1, q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # Actor
        new_action, log_prob = self.actor.sample(obs)
        q1_new, q2_new = self.critic(obs, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - q_new).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # Alpha — clamp to minimum to prevent exploration collapse
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        # Floor: never let alpha go below 0.01
        # Without this, alpha collapses to ~0.004 and SAC stops exploring
        # before discovering transport -> place
        with torch.no_grad():
            self.log_alpha.clamp_(min=np.log(0.01))

        # Soft update
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha":       self.alpha,
            "q1_mean":     q1.mean().item(),
        }

    def save(self, path):
        torch.save({
            "actor":         self.actor.state_dict(),
            "critic":        self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha":     self.log_alpha.data,
            "actor_opt":     self.actor_opt.state_dict(),
            "critic_opt":    self.critic_opt.state_dict(),
            "alpha_opt":     self.alpha_opt.state_dict(),
            "total_updates": self.total_updates,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.log_alpha.data = ckpt["log_alpha"]
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        self.alpha_opt.load_state_dict(ckpt["alpha_opt"])
        self.total_updates = ckpt["total_updates"]
        print(f"  Loaded checkpoint: {path} (updates={self.total_updates})")


#  Evaluation 

def evaluate(env, agent, n_episodes=30):
    results = {"reach": 0, "grasp": 0, "lift": 0, "place": 0, "push": 0}
    total_reward = 0
    ep_lengths = []
    max_heights = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        ep_r = 0
        reached = grasped = lifted = False

        for t in range(env.max_episode_steps):
            action = agent.select_action(obs, deterministic=True)
            obs, r, done, trunc, info = env.step(action)
            ep_r += r

            ee_pos = obs[0:3]
            fw     = obs[6]

            # Check each active ball
            for i in range(env.n_balls):
                offset = 7 + i * PER_BALL_DIM
                ball_pos  = obs[offset:offset+3]
                ball_rel  = obs[offset+3:offset+6]
                is_sorted = obs[offset+9]
                dist      = np.linalg.norm(ball_rel)

                if is_sorted > 0.5:
                    continue  # already placed

                if dist < 0.05:
                    reached = True
                if fw > 0.01 and fw < 0.06 and dist < 0.04:
                    grasped = True
                if ball_pos[2] > TABLE_Z + BALL_RADIUS + LIFT_THRESH:
                    lifted = True

            if done or trunc:
                break

        n_sorted = info.get("n_sorted", 0)
        n_in_bin = info.get("n_in_bin", 0)

        results["reach"] += int(reached)
        results["grasp"] += int(grasped)
        results["lift"]  += int(lifted)
        results["place"] += int(n_sorted >= env.n_balls)
        results["push"]  += int(n_in_bin > n_sorted)
        total_reward += ep_r
        ep_lengths.append(t + 1)
        max_heights.append(max(info.get("max_heights", [0.445])))

    rates = {k: v / n_episodes for k, v in results.items()}
    rates["reward"]     = total_reward / n_episodes
    rates["ep_len"]     = np.mean(ep_lengths)
    rates["max_ball_z"] = np.mean(max_heights)
    return rates


#  Logger 

class Logger:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self._csv_file = None
        self._csv_writer = None
        self._ep_rewards = deque(maxlen=100)

    def log_episode(self, episode, ep_reward, ep_len, total_steps, info):
        self._ep_rewards.append(ep_reward)

    def log_eval(self, total_steps, rates):
        row = {"step": total_steps, **rates}
        if self._csv_writer is None:
            self._csv_file = open(self.run_dir / "eval_log.csv", "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=row.keys())
            self._csv_writer.writeheader()
        self._csv_writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v
                                   for k, v in row.items()})
        self._csv_file.flush()

    def log_training(self, total_steps, episode, sac_logs):
        row = {"step": total_steps, "episode": episode,
               "mean_reward": np.mean(self._ep_rewards) if self._ep_rewards else 0,
               **sac_logs}
        csv_path = self.run_dir / "training_log.csv"
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v
                             for k, v in row.items()})

    @property
    def mean_reward(self):
        return np.mean(self._ep_rewards) if self._ep_rewards else 0

    def close(self):
        if self._csv_file:
            self._csv_file.close()


#  Config 

def get_config():
    p = argparse.ArgumentParser()
    p.add_argument("--total_steps",  type=int,   default=200_000)
    p.add_argument("--seed",         type=int,   default=0)
    p.add_argument("--n_balls",      type=int,   default=1)
    p.add_argument("--ep_length",    type=int,   default=200)
    p.add_argument("--resume",       type=str,   default=None)
    p.add_argument("--run_name",     type=str,   default=None)
    p.add_argument("--device",       type=str,   default=None)

    # SAC
    p.add_argument("--hidden",       type=int,   default=256)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--batch_size",   type=int,   default=256)
    p.add_argument("--buffer_size",  type=int,   default=200_000)
    p.add_argument("--warmup",       type=int,   default=1_000)
    p.add_argument("--update_every", type=int,   default=2)
    p.add_argument("--gamma",        type=float, default=0.99)

    # Logging
    p.add_argument("--log_every",    type=int,   default=2_000)
    p.add_argument("--eval_every",   type=int,   default=10_000)
    p.add_argument("--save_every",   type=int,   default=25_000)
    p.add_argument("--n_eval_eps",   type=int,   default=30)

    # Curriculum
    p.add_argument("--curriculum",   action="store_true",
                   help="Enable 1->2 ball curriculum")
    p.add_argument("--advance_threshold", type=float, default=0.30,
                   help="Place rate to advance to 2 balls")

    return p.parse_args()


#  Main 

def main():
    cfg = get_config()

    # Seed
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Device
    if cfg.device:
        device = cfg.device
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Run dir
    name = cfg.run_name or f"sac_v2_seed{cfg.seed}"
    run_dir = Path("runs") / name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(cfg), f, indent=2)

    print("=" * 60)
    print(f"SAC Training — PandaSortEnv v2")
    print(f"=" * 60)
    print(f"  Device:      {device}")
    print(f"  Run dir:     {run_dir}")
    print(f"  Total steps: {cfg.total_steps:,}")
    print(f"  N balls:     {cfg.n_balls}")
    print(f"  Episode len: {cfg.ep_length}")
    print(f"  Curriculum:  {cfg.curriculum}")

    # Env
    env = PandaSortEnv(
        n_balls=cfg.n_balls,
        reward_mode="dense",
        max_episode_steps=cfg.ep_length,
        seed=cfg.seed,
    )

    # Agent
    agent = SAC(
        obs_dim=OBS_DIM, act_dim=4, hidden=cfg.hidden,
        lr=cfg.lr, gamma=cfg.gamma, device=device,
    )
    total_params = sum(p.numel() for p in agent.actor.parameters()) + \
                   sum(p.numel() for p in agent.critic.parameters())
    print(f"  Parameters:  {total_params:,}")

    # Buffer
    buffer = ReplayBuffer(capacity=cfg.buffer_size, obs_dim=OBS_DIM)

    # Resume
    start_step = 0
    start_episode = 0
    if cfg.resume:
        agent.load(cfg.resume)
        # Try to extract step count from filename
        try:
            start_step = int(Path(cfg.resume).stem.split("_")[-1])
        except ValueError:
            pass
        print(f"  Resumed from step {start_step}")

    # Logger
    logger = Logger(run_dir)

    # Curriculum state
    curriculum_advanced = False

    #  Training loop 
    total_steps = start_step
    episode = start_episode
    best_place_rate = 0.0
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"  Starting training at step {total_steps}")
    print(f"{'='*60}\n")

    while total_steps < cfg.total_steps:
        obs, info = env.reset()
        ep_reward = 0.0
        episode += 1

        for t in range(env.max_episode_steps):
            # Action
            if total_steps < cfg.warmup + start_step:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs)

            next_obs, reward, done, trunc, info = env.step(action)
            buffer.add(obs, action, reward, next_obs, done)
            ep_reward += reward
            total_steps += 1
            obs = next_obs

            # Update
            if (total_steps >= cfg.warmup + start_step
                    and total_steps % cfg.update_every == 0
                    and buffer.size >= cfg.batch_size):
                sac_logs = agent.update(buffer, cfg.batch_size)

                # Log training stats
                if total_steps % cfg.log_every == 0:
                    logger.log_training(total_steps, episode, sac_logs)
                    elapsed = time.time() - t_start
                    sps = (total_steps - start_step) / elapsed
                    print(f"  Step {total_steps:7d} | "
                          f"Ep {episode:5d} | "
                          f"MeanR {logger.mean_reward:7.2f} | "
                          f"Alpha {agent.alpha:.4f} | "
                          f"Q1 {sac_logs['q1_mean']:7.2f} | "
                          f"SPS {sps:.0f}")

            # Eval
            if total_steps % cfg.eval_every == 0 and total_steps > start_step:
                rates = evaluate(env, agent, cfg.n_eval_eps)
                logger.log_eval(total_steps, rates)

                print(f"\n  {'='*50}")
                print(f"  EVAL @ {total_steps:,}")
                print(f"  {'='*50}")
                print(f"    Reach:  {rates['reach']:.0%}")
                print(f"    Grasp:  {rates['grasp']:.0%}")
                print(f"    Lift:   {rates['lift']:.0%}")
                print(f"    Place:  {rates['place']:.0%}  (pushed: {rates['push']:.0%})")
                print(f"    Reward: {rates['reward']:.2f}")
                print(f"    Ep len: {rates['ep_len']:.0f}")
                print(f"    Max ball z: {rates['max_ball_z']:.3f}  (need >0.525)")
                print(f"  {'='*50}\n")

                # Save best
                if rates['place'] > best_place_rate:
                    best_place_rate = rates['place']
                    agent.save(str(run_dir / "best_model.pt"))
                    print(f"    New best! Place rate: {best_place_rate:.0%}")

                # Curriculum: advance to 2 balls
                if (cfg.curriculum and not curriculum_advanced
                        and rates['place'] >= cfg.advance_threshold
                        and cfg.n_balls == 1):
                    curriculum_advanced = True
                    env.n_balls = 2
                    print(f"\n    *** CURRICULUM: Advancing to 2 balls ***\n")

            # Save checkpoint
            if total_steps % cfg.save_every == 0 and total_steps > start_step:
                agent.save(str(run_dir / f"checkpoint_{total_steps}.pt"))

            if done or trunc:
                break

        logger.log_episode(episode, ep_reward, t + 1, total_steps, info)

    #  Final 
    agent.save(str(run_dir / "final_model.pt"))

    print(f"\n{'='*60}")
    print("FINAL EVALUATION (50 episodes)")
    print(f"{'='*60}")
    rates = evaluate(env, agent, 50)
    print(f"  Reach:  {rates['reach']:.0%}")
    print(f"  Grasp:  {rates['grasp']:.0%}")
    print(f"  Lift:   {rates['lift']:.0%}")
    print(f"  Place:  {rates['place']:.0%}  (pushed: {rates['push']:.0%})")
    print(f"  Reward: {rates['reward']:.2f}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed/3600:.1f}h")
    print(f"  Steps/sec:  {(cfg.total_steps - start_step)/elapsed:.0f}")
    print(f"  Best place: {best_place_rate:.0%}")

    logger.close()
    env.close()


if __name__ == "__main__":
    main()