"""
Frozen VLM + SAC for language-conditioned ball sorting.

Architecture:
    Frozen SigLIP vision encoder -> 768D per camera (mean-pooled)
    Frozen LLaMA text encoder -> 960D (mean-pooled)
    Concat: vision(1536D) + language(960D) + proprio(7D) = 2503D
    Trainable SAC actor/critic on top (~2M params)

The VLM never receives gradients. SAC learns reactive control
from rich pretrained visual + language features.

Usage:
    python scripts/train_vlm_sac.py --n_balls 1 --total_steps 200000
    python scripts/train_vlm_sac.py --n_balls 1 --total_steps 200000 --resume runs/vlm_sac_1ball/best_model.pt
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
    N_MAX_BALLS, BALL_RADIUS, LIFT_THRESH, BIN_POSITIONS,
)

#  Constants 

IMG_H, IMG_W = 480, 640
VIS_DIM      = 768      # SigLIP hidden size per camera
LANG_DIM     = 960      # LLaMA text hidden size
PROPRIO_DIM  = 7        # ee_pos(3) + ee_vel(3) + finger_width(1)
FEAT_DIM     = VIS_DIM * 2 + LANG_DIM + PROPRIO_DIM  # 2503


#  Image rendering (same as vision SAC) 

class ImageRenderer:
    def __init__(self, model, data):
        self.model = model
        self.data  = data
        model.vis.global_.offwidth  = IMG_W
        model.vis.global_.offheight = IMG_H
        self.overhead_renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)
        self.wrist_renderer    = mujoco.Renderer(model, height=IMG_H, width=IMG_W)
        self._ee_site = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "ee_center_site")

    def render(self):
        """Returns (overhead, wrist) as float32 CHW [0,1] numpy arrays."""
        self.overhead_renderer.update_scene(self.data, camera="overhead_cam")
        overhead = self.overhead_renderer.render().copy()

        ee_pos = self.data.site_xpos[self._ee_site].copy()
        cam = mujoco.MjvCamera()
        cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = ee_pos - np.array([0, 0, 0.08])
        cam.distance  = 0.20
        cam.azimuth   = 180
        cam.elevation = -75
        self.wrist_renderer.update_scene(self.data, camera=cam)
        wrist = self.wrist_renderer.render().copy()

        return (
            overhead.transpose(2, 0, 1).astype(np.float32) / 255.0,
            wrist.transpose(2, 0, 1).astype(np.float32) / 255.0,
        )

    def close(self):
        self.overhead_renderer.close()
        self.wrist_renderer.close()


#  Frozen VLM Feature Extractor 

class VLMFeatureExtractor:
    """
    Extracts frozen features from SmolVLA's vision + language encoders.

    Vision: SigLIP (384x384 input, 576 patches x 768D) -> mean pool -> 768D
    Language: LLaMA text model (48 tokens x 960D) -> mean pool -> 960D

    All frozen. No gradients ever flow through this.
    """

    def __init__(self, checkpoint_path, device="mps"):
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from transformers import AutoTokenizer

        self.device = torch.device(device)

        print(f"Loading frozen VLM from {checkpoint_path}...")
        policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
        policy.to(self.device).float().eval()

        # Extract submodules
        vlm = policy.model.vlm_with_expert.vlm
        self.vision_model = vlm.model.vision_model
        self.text_model   = vlm.model.text_model
        self.connector    = vlm.model.connector  # might be useful later

        # Freeze everything
        for p in self.vision_model.parameters():
            p.requires_grad = False
        for p in self.text_model.parameters():
            p.requires_grad = False
        for p in self.connector.parameters():
            p.requires_grad = False

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

        # Image resize helper — SigLIP expects 384x384
        self._img_size = 384

        # Cache for language embeddings (same task string -> same embedding)
        self._lang_cache = {}

        print(f"  Vision: SigLIP -> {VIS_DIM}D per camera")
        print(f"  Language: LLaMA -> {LANG_DIM}D")
        print(f"  Total frozen features: {FEAT_DIM}D")

    @torch.no_grad()
    def encode_image(self, img_chw: np.ndarray) -> torch.Tensor:
        """
        img_chw: (3, H, W) float32 [0, 1]
        Returns: (768,) feature vector on device
        """
        img = torch.from_numpy(img_chw).unsqueeze(0).to(self.device)
        # Resize to 384x384
        img = F.interpolate(img, size=(self._img_size, self._img_size),
                           mode='bilinear', align_corners=False)
        # SigLIP expects [-1, 1]
        img = img * 2.0 - 1.0

        out = self.vision_model(img)
        # last_hidden_state: (1, 576, 768) -> mean pool -> (1, 768)
        feat = out.last_hidden_state.mean(dim=1).squeeze(0)
        return feat

    @torch.no_grad()
    def encode_language(self, task_str: str) -> torch.Tensor:
        """
        task_str: natural language instruction
        Returns: (960,) feature vector on device
        """
        if task_str in self._lang_cache:
            return self._lang_cache[task_str]

        tokens = self.tokenizer(
            task_str, return_tensors="pt",
            padding="max_length", max_length=48, truncation=True)
        ids  = tokens.input_ids.to(self.device)
        mask = tokens.attention_mask.to(self.device)

        emb = self.text_model.embed_tokens(ids)
        out = self.text_model(inputs_embeds=emb, attention_mask=mask)
        h   = out.last_hidden_state  # (1, 48, 960)

        # Mean pool over non-padding tokens
        mask_exp = mask.unsqueeze(-1).float()
        pooled = (h * mask_exp).sum(1) / mask_exp.sum(1)  # (1, 960)
        feat = pooled.squeeze(0)

        self._lang_cache[task_str] = feat
        return feat

    def extract_features(self, overhead_chw, wrist_chw, proprio, task_str):
        """
        Full feature extraction: 2 cameras + language + proprio -> 2503D

        Args:
            overhead_chw: (3, H, W) float32 [0, 1]
            wrist_chw:    (3, H, W) float32 [0, 1]
            proprio:      (7,) float32
            task_str:     string

        Returns: (2503,) numpy float32
        """
        oh_feat   = self.encode_image(overhead_chw)      # (768,)
        wr_feat   = self.encode_image(wrist_chw)          # (768,)
        lang_feat = self.encode_language(task_str)         # (960,)
        prop_t    = torch.from_numpy(proprio).float().to(self.device)  # (7,)

        feat = torch.cat([oh_feat, wr_feat, lang_feat, prop_t])  # (2503,)
        return feat.cpu().numpy()


#  Replay Buffer 

class ReplayBuffer:
    def __init__(self, capacity=200_000, feat_dim=FEAT_DIM, act_dim=4):
        self.capacity = capacity
        self.feat     = np.zeros((capacity, feat_dim), dtype=np.float32)
        self.action   = np.zeros((capacity, act_dim),  dtype=np.float32)
        self.reward   = np.zeros((capacity, 1),        dtype=np.float32)
        self.next_feat = np.zeros((capacity, feat_dim), dtype=np.float32)
        self.done     = np.zeros((capacity, 1),        dtype=np.float32)
        self._ptr  = 0
        self._size = 0

    def add(self, feat, action, reward, next_feat, done):
        i = self._ptr
        self.feat[i]      = feat
        self.action[i]    = action
        self.reward[i]    = reward
        self.next_feat[i] = next_feat
        self.done[i]      = float(done)
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size, device="cpu"):
        idx = np.random.randint(0, self._size, batch_size)
        return (
            torch.from_numpy(self.feat[idx]).to(device),
            torch.from_numpy(self.action[idx]).to(device),
            torch.from_numpy(self.reward[idx]).to(device),
            torch.from_numpy(self.next_feat[idx]).to(device),
            torch.from_numpy(self.done[idx]).to(device),
        )

    @property
    def size(self):
        return self._size


#  SAC Networks (same proven architecture) 

class Actor(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, act_dim=4, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.LayerNorm(hidden), nn.ReLU(),
        )
        self.mean    = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
        self._init_weights()

    def forward(self, feat):
        h = self.net(feat)
        return self.mean(h), self.log_std(h).clamp(-5, 2)

    def sample(self, feat):
        mean, log_std = self(feat)
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
    def __init__(self, feat_dim=FEAT_DIM, act_dim=4, hidden=512):
        super().__init__()
        dim = feat_dim + act_dim

        def make_q():
            return nn.Sequential(
                nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        self.q1 = make_q()
        self.q2 = make_q()

    def forward(self, feat, action):
        x = torch.cat([feat, action], -1)
        return self.q1(x), self.q2(x)


#  SAC Agent 

class VLM_SAC:
    def __init__(self, feat_dim=FEAT_DIM, act_dim=4, hidden=512,
                 lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2,
                 device="cpu"):
        self.device  = torch.device(device)
        self.gamma   = gamma
        self.tau     = tau
        self.act_dim = act_dim

        self.actor  = Actor(feat_dim, act_dim, hidden).to(self.device)
        self.critic = Critic(feat_dim, act_dim, hidden).to(self.device)
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
    def select_action(self, feat, deterministic=False):
        feat_t = torch.from_numpy(feat).float().unsqueeze(0).to(self.device)
        if deterministic:
            mean, _ = self.actor(feat_t)
            action = torch.tanh(mean)
        else:
            action, _ = self.actor.sample(feat_t)
        return action.squeeze(0).cpu().numpy()

    def update(self, buffer, batch_size=256):
        self.total_updates += 1
        feat, action, reward, next_feat, done = buffer.sample(batch_size, self.device)

        # Critic
        with torch.no_grad():
            next_action, next_lp = self.actor.sample(next_feat)
            q1_next, q2_next = self.critic_target(next_feat, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_lp
            q_target = reward + self.gamma * (1 - done) * q_next

        q1, q2 = self.critic(feat, action)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # Actor
        new_action, log_prob = self.actor.sample(feat)
        q1_new, q2_new = self.critic(feat, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - q_new).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # Alpha
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
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


#  Task string generation 

def generate_task_string(ball_colors, n_balls):
    if n_balls == 1:
        color = ball_colors[0]
        return f"pick up the {color} ball and place it in the {color} bin"
    else:
        color_list = " and ".join(
            [f"the {c} ball" for c in ball_colors[:n_balls]])
        return f"sort {color_list} into their matching bins"


#  Evaluation 

def evaluate(env, renderer, vlm, agent, n_episodes=20):
    results = {"reach": 0, "grasp": 0, "lift": 0, "place": 0, "push": 0}
    total_reward = 0

    for _ in range(n_episodes):
        obs, info = env.reset()
        ball_colors = info["ball_colors"]
        task = generate_task_string(ball_colors, env.n_balls)

        overhead, wrist = renderer.render()
        proprio = obs[0:7].astype(np.float32)
        feat = vlm.extract_features(overhead, wrist, proprio, task)

        ep_r = 0
        grasped = False

        for t in range(env.max_episode_steps):
            action = agent.select_action(feat, deterministic=True)
            obs, r, done, trunc, info = env.step(action)

            overhead, wrist = renderer.render()
            proprio = obs[0:7].astype(np.float32)
            feat = vlm.extract_features(overhead, wrist, proprio, task)
            ep_r += r

            fw = obs[6]
            for i in range(env.n_balls):
                offset = 7 + i * PER_BALL_DIM
                ball_rel = obs[offset + 3:offset + 6]
                dist = np.linalg.norm(ball_rel)
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
        total_reward += ep_r

    n = n_episodes
    return {k: v / n for k, v in results.items()} | {"reward": total_reward / n}


#  Config 

def get_config():
    p = argparse.ArgumentParser()
    p.add_argument("--vlm_checkpoint", type=str,
                   default="outputs/train/smolvla_panda_sort_v2/checkpoints/005000/pretrained_model")
    p.add_argument("--total_steps",  type=int,   default=200_000)
    p.add_argument("--seed",         type=int,   default=0)
    p.add_argument("--n_balls",      type=int,   default=1)
    p.add_argument("--ep_length",    type=int,   default=200)
    p.add_argument("--run_name",     type=str,   default=None)
    p.add_argument("--device",       type=str,   default=None)
    p.add_argument("--resume",       type=str,   default=None)

    # SAC
    p.add_argument("--hidden",       type=int,   default=512)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--batch_size",   type=int,   default=256)
    p.add_argument("--buffer_size",  type=int,   default=200_000)
    p.add_argument("--warmup",       type=int,   default=5_000)
    p.add_argument("--update_every", type=int,   default=2)
    p.add_argument("--gamma",        type=float, default=0.99)

    # Logging
    p.add_argument("--log_every",    type=int,   default=2_000)
    p.add_argument("--eval_every",   type=int,   default=10_000)
    p.add_argument("--save_every",   type=int,   default=25_000)
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

    name    = cfg.run_name or f"vlm_sac_{cfg.n_balls}ball_seed{cfg.seed}"
    run_dir = Path("runs") / name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(cfg), f, indent=2)

    print("=" * 60)
    print("Frozen VLM + SAC Training")
    print("=" * 60)
    print(f"  Device:      {device}")
    print(f"  Run dir:     {run_dir}")
    print(f"  Total steps: {cfg.total_steps:,}")
    print(f"  N balls:     {cfg.n_balls}")
    print(f"  Feature dim: {FEAT_DIM}")

    #  Env + Renderer 
    env = PandaSortEnv(
        n_balls=cfg.n_balls,
        reward_mode="dense",
        max_episode_steps=cfg.ep_length,
        seed=cfg.seed,
    )
    renderer = ImageRenderer(env.model, env.data)

    #  Frozen VLM 
    vlm = VLMFeatureExtractor(cfg.vlm_checkpoint, device=device)

    #  SAC Agent (trains on CPU — features are already extracted) 
    # SAC on CPU because features are numpy. MPS overhead not worth it
    # for small MLPs.
    sac_device = "cpu"
    agent = VLM_SAC(
        feat_dim=FEAT_DIM, act_dim=4, hidden=cfg.hidden,
        lr=cfg.lr, gamma=cfg.gamma, device=sac_device,
    )
    total_params = sum(p.numel() for p in agent.actor.parameters()) + \
                   sum(p.numel() for p in agent.critic.parameters())
    print(f"  SAC params:  {total_params:,}")

    buffer = ReplayBuffer(capacity=cfg.buffer_size, feat_dim=FEAT_DIM)

    # Resume
    start_step = 0
    if cfg.resume:
        agent.load(cfg.resume)
        try:
            start_step = int(Path(cfg.resume).stem.split("_")[-1])
        except ValueError:
            # best_model.pt or final_model.pt — skip warmup, start from 0
            # but set start_step to warmup value so random phase is skipped
            start_step = cfg.warmup
        print(f"  Resumed from step {start_step}")

    #  Timing benchmark 
    print("\n  Benchmarking VLM feature extraction...")
    obs_bench, info_bench = env.reset()
    oh_bench, wr_bench = renderer.render()
    proprio_bench = obs_bench[0:7].astype(np.float32)
    task_bench = "pick up the red ball and place it in the red bin"

    t0 = time.time()
    for _ in range(10):
        _ = vlm.extract_features(oh_bench, wr_bench, proprio_bench, task_bench)
    feat_time = (time.time() - t0) / 10
    print(f"  Feature extraction: {feat_time*1000:.0f}ms/step")
    print(f"  Estimated wall time: {cfg.total_steps * feat_time / 3600:.1f}h")

    #  Training loop 
    total_steps     = start_step
    best_place_rate = 0.0
    ep_rewards      = deque(maxlen=100)
    t_start         = time.time()
    episode         = 0

    csv_path   = run_dir / "eval_log.csv"
    csv_file   = None
    csv_writer = None

    print(f"\n  Starting training at step {total_steps}\n")

    while total_steps < cfg.total_steps:
        obs, info = env.reset()
        ball_colors = info["ball_colors"]
        task = generate_task_string(ball_colors, env.n_balls)

        overhead, wrist = renderer.render()
        proprio = obs[0:7].astype(np.float32)
        feat = vlm.extract_features(overhead, wrist, proprio, task)

        ep_reward = 0.0
        episode += 1

        for t in range(env.max_episode_steps):
            # Action
            if total_steps < cfg.warmup + start_step:
                action = env.action_space.sample()
            else:
                action = agent.select_action(feat)

            next_obs, reward, done, trunc, info = env.step(action)

            next_overhead, next_wrist = renderer.render()
            next_proprio = next_obs[0:7].astype(np.float32)
            next_feat = vlm.extract_features(
                next_overhead, next_wrist, next_proprio, task)

            buffer.add(feat, action, reward, next_feat, done)
            ep_reward   += reward
            total_steps += 1
            feat = next_feat
            obs  = next_obs

            # Update SAC
            if (total_steps >= cfg.warmup + start_step
                    and total_steps % cfg.update_every == 0
                    and buffer.size >= cfg.batch_size):
                sac_logs = agent.update(buffer, cfg.batch_size)

                if total_steps % cfg.log_every == 0:
                    elapsed = time.time() - t_start
                    sps = (total_steps - start_step) / elapsed
                    mean_r = np.mean(ep_rewards) if ep_rewards else 0
                    print(f"  Step {total_steps:7d} | "
                          f"Ep {episode:5d} | "
                          f"MeanR {mean_r:7.2f} | "
                          f"Alpha {agent.alpha:.4f} | "
                          f"Q1 {sac_logs['q1_mean']:7.2f} | "
                          f"SPS {sps:.1f}")

            # Eval
            if total_steps % cfg.eval_every == 0 and total_steps > start_step:
                rates = evaluate(env, renderer, vlm, agent, cfg.n_eval_eps)

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

            # Save checkpoint
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
    rates = evaluate(env, renderer, vlm, agent, 50)
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