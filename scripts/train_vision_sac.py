"""
Vision SAC training on PandaSortEnv v2 — with DrQ-style augmentation.

Replaces 51D state obs with:
    - Overhead camera (64x64x3)
    - Wrist camera (64x64x3)
    - Proprio (7D: ee_pos, ee_vel, finger_width)

Key stabilization:
    - Render at 68x68, random crop to 64x64 during training
    - Shared encoder between actor and critic
    - Staged warmup: critic-only -> encoder unfreezes -> actor unfreezes
    - Encoder updated ONLY through critic loss (actor uses detached features)
    - Separate optimizers: encoder (1e-5), critic heads (1e-4), actor (3e-4)

Usage:
    python scripts/train_vision_sac.py --total_steps 500000
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
import sys
import mujoco

sys.path.append(str(Path(__file__).parent.parent))
from env.panda_sort_env import (
    PandaSortEnv, TABLE_Z, OBS_DIM, PER_BALL_DIM,
    N_MAX_BALLS, BALL_RADIUS, LIFT_THRESH
)
from policy.cnn_encoder import VisionEncoder, preprocess_image, random_crop


#  Constants 

RENDER_SIZE = 68
CROP_SIZE   = 64


#  Image rendering helper 

class ImageRenderer:
    def __init__(self, model, data, img_size=RENDER_SIZE):
        self.model    = model
        self.data     = data
        self.img_size = img_size

        model.vis.global_.offwidth  = img_size
        model.vis.global_.offheight = img_size

        self.overhead_renderer = mujoco.Renderer(model, height=img_size, width=img_size)
        self.wrist_renderer    = mujoco.Renderer(model, height=img_size, width=img_size)

        self._ee_site = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "ee_center_site")

    def render(self):
        self.overhead_renderer.update_scene(self.data, camera="overhead_cam")
        overhead = self.overhead_renderer.render()

        ee_pos = self.data.site_xpos[self._ee_site].copy()
        cam = mujoco.MjvCamera()
        cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = ee_pos - np.array([0, 0, 0.08])
        cam.distance  = 0.20
        cam.azimuth   = 180
        cam.elevation = -75
        self.wrist_renderer.update_scene(self.data, camera=cam)
        wrist = self.wrist_renderer.render()

        return (
            preprocess_image(overhead, self.img_size),
            preprocess_image(wrist,    self.img_size),
        )

    def close(self):
        self.overhead_renderer.close()
        self.wrist_renderer.close()


#  Vision Replay Buffer 

class VisionReplayBuffer:
    def __init__(self, capacity=100_000, img_size=RENDER_SIZE, proprio_dim=7, act_dim=4):
        self.capacity = capacity
        h, w = img_size, img_size

        self.overhead      = np.zeros((capacity, h, w, 3), dtype=np.uint8)
        self.wrist         = np.zeros((capacity, h, w, 3), dtype=np.uint8)
        self.proprio       = np.zeros((capacity, proprio_dim), dtype=np.float32)
        self.action        = np.zeros((capacity, act_dim),     dtype=np.float32)
        self.reward        = np.zeros((capacity, 1),           dtype=np.float32)
        self.next_overhead = np.zeros((capacity, h, w, 3),     dtype=np.uint8)
        self.next_wrist    = np.zeros((capacity, h, w, 3),     dtype=np.uint8)
        self.next_proprio  = np.zeros((capacity, proprio_dim), dtype=np.float32)
        self.done          = np.zeros((capacity, 1),           dtype=np.float32)

        self._ptr  = 0
        self._size = 0

    def add(self, overhead, wrist, proprio, action, reward,
            next_overhead, next_wrist, next_proprio, done):
        i = self._ptr
        self.overhead[i]      = (overhead.transpose(1, 2, 0) * 255).astype(np.uint8)
        self.wrist[i]         = (wrist.transpose(1, 2, 0) * 255).astype(np.uint8)
        self.proprio[i]       = proprio
        self.action[i]        = action
        self.reward[i]        = reward
        self.next_overhead[i] = (next_overhead.transpose(1, 2, 0) * 255).astype(np.uint8)
        self.next_wrist[i]    = (next_wrist.transpose(1, 2, 0) * 255).astype(np.uint8)
        self.next_proprio[i]  = next_proprio
        self.done[i]          = float(done)
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size, device="cpu"):
        idx = np.random.randint(0, self._size, batch_size)

        def to_img_tensor(arr):
            return torch.from_numpy(
                arr.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            ).to(device)

        return (
            to_img_tensor(self.overhead[idx]),
            to_img_tensor(self.wrist[idx]),
            torch.from_numpy(self.proprio[idx]).to(device),
            torch.from_numpy(self.action[idx]).to(device),
            torch.from_numpy(self.reward[idx]).to(device),
            to_img_tensor(self.next_overhead[idx]),
            to_img_tensor(self.next_wrist[idx]),
            torch.from_numpy(self.next_proprio[idx]).to(device),
            torch.from_numpy(self.done[idx]).to(device),
        )

    @property
    def size(self):
        return self._size


#  Vision SAC Networks 

class VisionActor(nn.Module):
    def __init__(self, encoder, act_dim=4, hidden=256):
        super().__init__()
        self.encoder = encoder
        self.net = nn.Sequential(
            nn.Linear(encoder.out_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),           nn.LayerNorm(hidden), nn.ReLU(),
        )
        self.mean    = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, overhead, wrist, proprio):
        feat = self.encoder(overhead, wrist, proprio)
        h    = self.net(feat)
        return self.mean(h), self.log_std(h).clamp(-5, 2)

    def sample(self, overhead, wrist, proprio):
        mean, log_std = self(overhead, wrist, proprio)
        std  = log_std.exp()
        x    = mean + torch.randn_like(mean) * std
        action   = torch.tanh(x)
        log_prob = (
            -0.5 * ((x - mean) / (std + 1e-8)).pow(2)
            - log_std - 0.5 * np.log(2 * np.pi)
        ).sum(-1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(-1, keepdim=True)
        return action, log_prob


class VisionCritic(nn.Module):
    def __init__(self, encoder, act_dim=4, hidden=256):
        super().__init__()
        self.encoder = encoder
        dim = encoder.out_dim + act_dim

        def make_q():
            return nn.Sequential(
                nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        self.q1 = make_q()
        self.q2 = make_q()

    def forward(self, overhead, wrist, proprio, action):
        feat = self.encoder(overhead, wrist, proprio)
        x    = torch.cat([feat, action], -1)
        return self.q1(x), self.q2(x)


#  Vision SAC Agent 

class VisionSAC:
    def __init__(self, per_cam_dim=128, proprio_dim=7, act_dim=4, hidden=256,
                 lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2,
                 encoder_lr=1e-5, critic_lr=1e-4, device="cpu"):
        self.device  = torch.device(device)
        self.gamma   = gamma
        self.tau     = tau
        self.act_dim = act_dim

        self.encoder = VisionEncoder(per_cam_dim, proprio_dim).to(self.device)
        self.actor   = VisionActor(self.encoder, act_dim, hidden).to(self.device)
        self.critic  = VisionCritic(self.encoder, act_dim, hidden).to(self.device)

        self.critic_target = deepcopy(self.critic).to(self.device)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.log_alpha = torch.tensor(
            np.log(alpha), requires_grad=True,
            dtype=torch.float32, device=self.device)
        self.target_entropy = -float(act_dim)

        # Three separate optimizers — clean gradient routing
        self.encoder_opt = torch.optim.Adam(
            self.encoder.parameters(), lr=encoder_lr)
        self.critic_opt  = torch.optim.Adam(
            list(self.critic.q1.parameters()) +
            list(self.critic.q2.parameters()), lr=critic_lr)
        self.actor_opt   = torch.optim.Adam(
            list(self.actor.net.parameters()) +
            list(self.actor.mean.parameters()) +
            list(self.actor.log_std.parameters()), lr=lr)
        self.alpha_opt   = torch.optim.Adam([self.log_alpha], lr=lr)

        self.total_updates = 0

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    @torch.no_grad()
    def select_action(self, overhead, wrist, proprio, deterministic=False):
        oh  = torch.from_numpy(overhead).float().unsqueeze(0).to(self.device)
        wr  = torch.from_numpy(wrist).float().unsqueeze(0).to(self.device)
        pad = (RENDER_SIZE - CROP_SIZE) // 2
        oh  = oh[:, :, pad:pad+CROP_SIZE, pad:pad+CROP_SIZE]
        wr  = wr[:, :, pad:pad+CROP_SIZE, pad:pad+CROP_SIZE]
        pr  = torch.from_numpy(proprio).float().unsqueeze(0).to(self.device)
        if deterministic:
            mean, _ = self.actor(oh, wr, pr)
            action  = torch.tanh(mean)
        else:
            action, _ = self.actor.sample(oh, wr, pr)
        return action.squeeze(0).cpu().numpy()

    def _augment(self, oh, wr):
        return random_crop(oh, CROP_SIZE), random_crop(wr, CROP_SIZE)

    def update(self, buffer, batch_size=128, freeze_encoder=False, freeze_actor=False):
        self.total_updates += 1
        (oh, wr, pr, action, reward,
         next_oh, next_wr, next_pr, done) = buffer.sample(batch_size, self.device)

        oh_aug1, wr_aug1     = self._augment(oh, wr)
        oh_aug2, wr_aug2     = self._augment(oh, wr)
        next_oh_a, next_wr_a = self._augment(next_oh, next_wr)

        #  Critic update 
        with torch.no_grad():
            next_a, next_lp = self.actor.sample(next_oh_a, next_wr_a, next_pr)
            q1_n, q2_n      = self.critic_target(next_oh_a, next_wr_a, next_pr, next_a)
            q_next   = torch.min(q1_n, q2_n) - self.alpha * next_lp
            q_target = reward + self.gamma * (1 - done) * q_next

        q1_a, q2_a = self.critic(oh_aug1, wr_aug1, pr, action)
        q1_b, q2_b = self.critic(oh_aug2, wr_aug2, pr, action)
        q1 = (q1_a + q1_b) / 2
        q2 = (q2_a + q2_b) / 2
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_opt.zero_grad()
        if not freeze_encoder:
            self.encoder_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.critic.q1.parameters()) +
            list(self.critic.q2.parameters()), 1.0)
        self.critic_opt.step()
        if not freeze_encoder:
            nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
            self.encoder_opt.step()

        #  Actor update 
        actor_loss_val = 0.0
        if not freeze_actor:
            oh_aug3, wr_aug3 = self._augment(oh, wr)

            with torch.no_grad():
                feat = self.encoder(oh_aug3, wr_aug3, pr)

            h       = self.actor.net(feat)
            mean    = self.actor.mean(h)
            log_std = self.actor.log_std(h).clamp(-5, 2)
            std     = log_std.exp()
            x       = mean + torch.randn_like(mean) * std
            new_act = torch.tanh(x)
            log_prob = (
                -0.5 * ((x - mean) / (std + 1e-8)).pow(2)
                - log_std - 0.5 * np.log(2 * np.pi)
            ).sum(-1, keepdim=True)
            log_prob -= torch.log(1 - new_act.pow(2) + 1e-6).sum(-1, keepdim=True)

            q_in   = torch.cat([feat, new_act], -1)
            q1_new = self.critic.q1(q_in)
            q2_new = self.critic.q2(q_in)
            actor_loss = (self.alpha * log_prob - torch.min(q1_new, q2_new)).mean()
            actor_loss_val = actor_loss.item()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.net.parameters()) +
                list(self.actor.mean.parameters()) +
                list(self.actor.log_std.parameters()), 1.0)
            self.actor_opt.step()

            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            with torch.no_grad():
                self.log_alpha.clamp_(min=np.log(0.01))

        # Soft update target
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return {
            "critic_loss":    critic_loss.item(),
            "actor_loss":     actor_loss_val,
            "alpha":          self.alpha,
            "q1_mean":        q1.mean().item(),
            "freeze_encoder": int(freeze_encoder),
            "freeze_actor":   int(freeze_actor),
        }

    def save(self, path):
        torch.save({
            "encoder":       self.encoder.state_dict(),
            "actor":         self.actor.state_dict(),
            "critic":        self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha":     self.log_alpha.data,
            "encoder_opt":   self.encoder_opt.state_dict(),
            "critic_opt":    self.critic_opt.state_dict(),
            "actor_opt":     self.actor_opt.state_dict(),
            "alpha_opt":     self.alpha_opt.state_dict(),
            "total_updates": self.total_updates,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.log_alpha.data = ckpt["log_alpha"]
        self.encoder_opt.load_state_dict(ckpt["encoder_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.alpha_opt.load_state_dict(ckpt["alpha_opt"])
        self.total_updates = ckpt["total_updates"]
        print(f"  Loaded vision SAC from {path} (updates={self.total_updates})")


#  Helper 

def get_proprio(obs):
    return obs[0:7].copy()


#  Evaluation 

def evaluate(env, renderer, agent, n_episodes=20):
    results = {"reach": 0, "grasp": 0, "lift": 0, "place": 0, "push": 0}
    total_reward = 0

    for _ in range(n_episodes):
        obs, info       = env.reset()
        overhead, wrist = renderer.render()
        proprio         = get_proprio(obs)
        ep_r            = 0
        grasped         = False

        for t in range(env.max_episode_steps):
            action = agent.select_action(overhead, wrist, proprio, deterministic=True)
            obs, r, done, trunc, info = env.step(action)
            overhead, wrist = renderer.render()
            proprio = get_proprio(obs)
            ep_r   += r

            fw = proprio[6]
            for i in range(env.n_balls):
                offset   = 7 + i * PER_BALL_DIM
                ball_rel = obs[offset + 3:offset + 6]
                dist     = np.linalg.norm(ball_rel)
                if dist < 0.04 and fw > 0.01 and fw < 0.06:
                    grasped = True

            if done or trunc:
                break

        n_sorted = info.get("n_sorted", 0)
        n_in_bin = info.get("n_in_bin", 0)
        max_h    = max(info.get("max_heights", [0]))

        results["reach"] += int(max_h > TABLE_Z + BALL_RADIUS + 0.02)
        results["grasp"] += int(grasped)
        results["lift"]  += int(max_h > TABLE_Z + BALL_RADIUS + LIFT_THRESH)
        results["place"] += int(n_sorted >= env.n_balls)
        results["push"]  += int(n_in_bin > n_sorted)
        total_reward     += ep_r

    n = n_episodes
    return {k: v / n for k, v in results.items()} | {"reward": total_reward / n}


#  Config 

def get_config():
    p = argparse.ArgumentParser()
    p.add_argument("--total_steps",  type=int,   default=500_000)
    p.add_argument("--seed",         type=int,   default=0)
    p.add_argument("--n_balls",      type=int,   default=1)
    p.add_argument("--ep_length",    type=int,   default=200)
    p.add_argument("--run_name",     type=str,   default=None)
    p.add_argument("--device",       type=str,   default=None)
    p.add_argument("--resume",       type=str,   default=None)

    p.add_argument("--hidden",       type=int,   default=256)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--encoder_lr",   type=float, default=1e-5)
    p.add_argument("--critic_lr",    type=float, default=1e-4)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--buffer_size",  type=int,   default=100_000)
    p.add_argument("--warmup",       type=int,   default=5_000)
    p.add_argument("--update_every", type=int,   default=2)
    p.add_argument("--gamma",        type=float, default=0.99)

    p.add_argument("--freeze_encoder_steps", type=int, default=5_000)
    p.add_argument("--freeze_actor_steps",   type=int, default=2_000)

    p.add_argument("--log_every",    type=int,   default=2_000)
    p.add_argument("--eval_every",   type=int,   default=10_000)
    p.add_argument("--save_every",   type=int,   default=50_000)
    p.add_argument("--n_eval_eps",   type=int,   default=20)

    return p.parse_args()


#  Main 

def main():
    cfg = get_config()

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.device:
        device = cfg.device
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    name    = cfg.run_name or f"vision_sac_seed{cfg.seed}"
    run_dir = Path("runs") / name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(cfg), f, indent=2)

    print("=" * 60)
    print("Vision SAC Training — DrQ + staged warmup")
    print("=" * 60)
    print(f"  Device:              {device}")
    print(f"  Run dir:             {run_dir}")
    print(f"  Total steps:         {cfg.total_steps:,}")
    print(f"  Render size:         {RENDER_SIZE}x{RENDER_SIZE} -> crop {CROP_SIZE}x{CROP_SIZE}")
    print(f"  Warmup (random):     {cfg.warmup:,}")
    print(f"  Freeze encoder for:  {cfg.freeze_encoder_steps:,} post-warmup updates")
    print(f"  Freeze actor for:    {cfg.freeze_actor_steps:,} post-warmup updates")
    print(f"  LR actor/critic/enc: {cfg.lr} / {cfg.critic_lr} / {cfg.encoder_lr}")

    env = PandaSortEnv(
        n_balls=cfg.n_balls,
        reward_mode="dense",
        max_episode_steps=cfg.ep_length,
        seed=cfg.seed,
    )

    renderer = ImageRenderer(env.model, env.data, img_size=RENDER_SIZE)

    agent = VisionSAC(
        per_cam_dim=128, proprio_dim=7, act_dim=4, hidden=cfg.hidden,
        lr=cfg.lr, encoder_lr=cfg.encoder_lr, critic_lr=cfg.critic_lr,
        gamma=cfg.gamma, device=device,
    )
    total_params = sum(p.numel() for p in agent.actor.parameters()) + \
                   sum(p.numel() for p in agent.critic.parameters())
    print(f"  Parameters:          {total_params:,}")

    buffer = VisionReplayBuffer(capacity=cfg.buffer_size, img_size=RENDER_SIZE)

    start_step = 0
    if cfg.resume:
        agent.load(cfg.resume)
        try:
            start_step = int(Path(cfg.resume).stem.split("_")[-1])
        except ValueError:
            pass
        print(f"  Resumed from step {start_step}")

    total_steps     = start_step
    best_place_rate = 0.0
    ep_rewards      = deque(maxlen=100)
    t_start         = time.time()
    updates_done    = 0

    csv_path   = run_dir / "eval_log.csv"
    csv_file   = None
    csv_writer = None

    print(f"\n  Starting training at step {total_steps}\n")

    while total_steps < cfg.total_steps:
        obs, info       = env.reset()
        overhead, wrist = renderer.render()
        proprio         = get_proprio(obs)
        ep_reward       = 0.0

        for t in range(env.max_episode_steps):
            if total_steps < cfg.warmup + start_step:
                action = env.action_space.sample()
            else:
                action = agent.select_action(overhead, wrist, proprio)

            next_obs, reward, done, trunc, info = env.step(action)
            next_overhead, next_wrist = renderer.render()
            next_proprio = get_proprio(next_obs)

            buffer.add(overhead, wrist, proprio, action, reward,
                       next_overhead, next_wrist, next_proprio, done)

            ep_reward  += reward
            total_steps += 1
            overhead, wrist, proprio = next_overhead, next_wrist, next_proprio
            obs = next_obs

            if (total_steps >= cfg.warmup + start_step
                    and total_steps % cfg.update_every == 0
                    and buffer.size >= cfg.batch_size):

                freeze_enc = updates_done < cfg.freeze_encoder_steps
                freeze_act = updates_done < cfg.freeze_actor_steps
                sac_logs   = agent.update(buffer, cfg.batch_size,
                                          freeze_encoder=freeze_enc,
                                          freeze_actor=freeze_act)
                updates_done += 1

                if total_steps % cfg.log_every == 0:
                    elapsed = time.time() - t_start
                    sps     = (total_steps - start_step) / elapsed
                    mean_r  = np.mean(ep_rewards) if ep_rewards else 0
                    stage = ("critic-only" if freeze_enc else
                             "enc+critic " if freeze_act else
                             "full       ")
                    print(f"  Step {total_steps:7d} | "
                          f"MeanR {mean_r:7.2f} | "
                          f"Alpha {agent.alpha:.4f} | "
                          f"Q1 {sac_logs['q1_mean']:7.2f} | "
                          f"Stage [{stage}] | "
                          f"SPS {sps:.0f}")

            if total_steps % cfg.eval_every == 0 and total_steps > start_step:
                rates = evaluate(env, renderer, agent, cfg.n_eval_eps)

                print(f"\n  {'='*50}")
                print(f"  EVAL @ {total_steps:,}")
                print(f"  {'='*50}")
                print(f"    Reach:  {rates['reach']:.0%}")
                print(f"    Grasp:  {rates['grasp']:.0%}")
                print(f"    Lift:   {rates['lift']:.0%}")
                print(f"    Place:  {rates['place']:.0%}  (pushed: {rates['push']:.0%})")
                print(f"    Reward: {rates['reward']:.2f}")
                print(f"  {'='*50}\n")

                row = {"step": total_steps,
                       **{k: f"{v:.4f}" for k, v in rates.items()}}
                if csv_writer is None:
                    csv_file   = open(csv_path, "w", newline="")
                    csv_writer = csv.DictWriter(csv_file, fieldnames=row.keys())
                    csv_writer.writeheader()
                csv_writer.writerow(row)
                csv_file.flush()

                if rates["place"] > best_place_rate:
                    best_place_rate = rates["place"]
                    agent.save(str(run_dir / "best_model.pt"))
                    print(f"    New best! Place rate: {best_place_rate:.0%}")

            if total_steps % cfg.save_every == 0 and total_steps > start_step:
                agent.save(str(run_dir / f"checkpoint_{total_steps}.pt"))

            if done or trunc:
                break

        ep_rewards.append(ep_reward)

    #  Final 
    agent.save(str(run_dir / "final_model.pt"))

    print(f"\n{'='*60}")
    print("FINAL EVALUATION (50 episodes)")
    print(f"{'='*60}")
    rates = evaluate(env, renderer, agent, 50)
    print(f"  Reach:  {rates['reach']:.0%}")
    print(f"  Grasp:  {rates['grasp']:.0%}")
    print(f"  Lift:   {rates['lift']:.0%}")
    print(f"  Place:  {rates['place']:.0%}  (pushed: {rates['push']:.0%})")
    print(f"  Reward: {rates['reward']:.2f}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed/3600:.1f}h")
    print(f"  Best place: {best_place_rate:.0%}")

    if csv_file:
        csv_file.close()
    renderer.close()
    env.close()


if __name__ == "__main__":
    main()