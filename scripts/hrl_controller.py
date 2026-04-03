"""
Hierarchical RL Controller for multi-ball sorting.

Architecture:
    High-level policy (learned):
        - Observes all balls (51D env obs)
        - Outputs: ball index to target (discrete action, 0..n_balls-1)
        - Runs ONCE per sub-episode (when current ball is sorted or timeout)

    Low-level policy (frozen 1-ball checkpoint):
        - Receives remapped obs: target ball in slot 0, others zeroed
        - Outputs: 4D continuous action [dx, dy, dz, gripper]
        - Runs every env step (10Hz)

The key insight: test_2ball_sequential.py proved 95% success with
manual ball selection. We just need to learn the selection policy.

Usage:
    # Train high-level policy with frozen low-level
    python scripts/train_hrl.py --low_level_ckpt runs/sac_1ball_2/best_model.pt

    # Test
    python scripts/train_hrl.py --test --checkpoint runs/hrl/best_model.pt
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from env.panda_sort_env import (
    PandaSortEnv, OBS_DIM, PER_BALL_DIM, N_MAX_BALLS,
    TABLE_Z, BALL_RADIUS, LIFT_THRESH
)


#  Obs remapping (proven in test_2ball_sequential.py) 

def make_1ball_obs(full_obs: np.ndarray, target_ball_idx: int) -> np.ndarray:
    """
    Remap multi-ball obs so target ball is in slot 0, others zeroed.
    This is EXACTLY the trick from test_2ball_sequential.py that got 95%.
    """
    fake_obs = np.zeros(OBS_DIM, dtype=np.float32)
    fake_obs[0:7] = full_obs[0:7]  # robot state

    src_offset = 7 + target_ball_idx * PER_BALL_DIM
    dst_offset = 7  # slot 0
    fake_obs[dst_offset:dst_offset + PER_BALL_DIM] = \
        full_obs[src_offset:src_offset + PER_BALL_DIM]

    return fake_obs


#  High-level policy (DQN — discrete ball selection) 

class HighLevelPolicy(nn.Module):
    """
    Tiny DQN that picks which ball to target next.

    Input: compact high-level state (see extract_hl_obs)
    Output: Q-value per ball slot

    We use DQN not SAC because the action space is discrete
    (pick ball 0, 1, 2, or 3).
    """

    def __init__(self, obs_dim: int, n_actions: int = N_MAX_BALLS, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        self.n_actions = n_actions

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def select_action(self, obs: np.ndarray, epsilon: float = 0.0,
                      valid_mask: np.ndarray = None) -> int:
        """
        Epsilon-greedy action selection.
        valid_mask: boolean array, True for balls that are active AND unsorted.
        """
        if np.random.random() < epsilon:
            # Random among valid balls only
            if valid_mask is not None:
                valid_indices = np.where(valid_mask)[0]
                if len(valid_indices) > 0:
                    return np.random.choice(valid_indices)
            return np.random.randint(self.n_actions)

        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            q_values = self.forward(obs_t).squeeze(0).numpy()

            # Mask invalid balls with -inf
            if valid_mask is not None:
                q_values[~valid_mask] = -np.inf

            return int(np.argmax(q_values))


def extract_hl_obs(full_obs: np.ndarray, n_balls: int) -> np.ndarray:
    """
    Extract compact high-level observation for the ball selector.

    Per ball: pos(3) + dist_to_ee(1) + dist_to_bin(1) + sorted(1) + active(1) = 7
    Plus: ee_pos(3) + finger_width(1) = 4
    Total: 4 + n_max_balls * 7 = 4 + 28 = 32
    """
    ee_pos = full_obs[0:3]
    fw = full_obs[6]

    hl_obs = np.zeros(4 + N_MAX_BALLS * 7, dtype=np.float32)
    hl_obs[0:3] = ee_pos
    hl_obs[3] = fw

    for i in range(N_MAX_BALLS):
        offset_src = 7 + i * PER_BALL_DIM
        offset_dst = 4 + i * 7

        ball_pos = full_obs[offset_src:offset_src + 3]
        ball_rel = full_obs[offset_src + 3:offset_src + 6]
        bin_pos = full_obs[offset_src + 6:offset_src + 9]
        is_sorted = full_obs[offset_src + 9]
        is_active = full_obs[offset_src + 10]

        dist_ee = np.linalg.norm(ball_rel)
        dist_bin = np.linalg.norm(ball_pos[:2] - bin_pos[:2])

        hl_obs[offset_dst + 0:offset_dst + 3] = ball_pos
        hl_obs[offset_dst + 3] = dist_ee
        hl_obs[offset_dst + 4] = dist_bin
        hl_obs[offset_dst + 5] = is_sorted
        hl_obs[offset_dst + 6] = is_active

    return hl_obs


def get_valid_mask(full_obs: np.ndarray) -> np.ndarray:
    """Returns boolean mask: True for balls that are active AND unsorted."""
    mask = np.zeros(N_MAX_BALLS, dtype=bool)
    for i in range(N_MAX_BALLS):
        offset = 7 + i * PER_BALL_DIM
        is_active = full_obs[offset + 10] > 0.5
        is_sorted = full_obs[offset + 9] > 0.5
        mask[i] = is_active and not is_sorted
    return mask


#  DQN Replay Buffer 

class HLReplayBuffer:
    """Simple replay buffer for high-level DQN transitions."""

    def __init__(self, capacity: int = 50_000, obs_dim: int = 32):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.valid_mask = np.zeros((capacity, N_MAX_BALLS), dtype=bool)
        self._ptr = 0
        self._size = 0

    def add(self, obs, action, reward, next_obs, done, valid_mask):
        i = self._ptr
        self.obs[i] = obs
        self.action[i] = action
        self.reward[i] = reward
        self.next_obs[i] = next_obs
        self.done[i] = float(done)
        self.valid_mask[i] = valid_mask
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self._size, batch_size)
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.action[idx]),
            torch.from_numpy(self.reward[idx]),
            torch.from_numpy(self.next_obs[idx]),
            torch.from_numpy(self.done[idx]),
            self.valid_mask[idx],
        )

    @property
    def size(self):
        return self._size


#  Hierarchical Controller 

class HierarchicalController:
    """
    Combines high-level ball selector with frozen low-level motor policy.

    The high-level runs once when:
        - Episode starts
        - Current target ball gets sorted
        - Low-level times out (max_steps_per_ball exceeded)

    The low-level runs every env step, receiving remapped 1-ball obs.
    """

    def __init__(self, low_level_agent, n_balls: int = 2,
                 max_steps_per_ball: int = 150):
        self.low_level = low_level_agent
        self.n_balls = n_balls
        self.max_steps_per_ball = max_steps_per_ball

        # High-level DQN
        self.hl_obs_dim = 4 + N_MAX_BALLS * 7  # 32
        self.hl_policy = HighLevelPolicy(self.hl_obs_dim, N_MAX_BALLS)
        self.hl_target = HighLevelPolicy(self.hl_obs_dim, N_MAX_BALLS)
        self.hl_target.load_state_dict(self.hl_policy.state_dict())

        self.hl_optimizer = torch.optim.Adam(self.hl_policy.parameters(), lr=1e-3)
        self.hl_buffer = HLReplayBuffer(capacity=50_000, obs_dim=self.hl_obs_dim)

        self.gamma = 0.99
        self.target_update_freq = 100
        self.hl_updates = 0

        # Episode state
        self.current_target = None
        self.steps_on_target = 0

    def reset(self):
        self.current_target = None
        self.steps_on_target = 0

    def needs_new_target(self, full_obs: np.ndarray) -> bool:
        """Check if high-level needs to pick a new ball."""
        if self.current_target is None:
            return True

        # Current target got sorted
        offset = 7 + self.current_target * PER_BALL_DIM
        if full_obs[offset + 9] > 0.5:  # sorted flag
            return True

        # Timeout on current ball
        if self.steps_on_target >= self.max_steps_per_ball:
            return True

        # Current target not active (shouldn't happen but safety)
        if full_obs[offset + 10] < 0.5:
            return True

        return False

    def select_target(self, full_obs: np.ndarray, epsilon: float = 0.0) -> int:
        """High-level selects which ball to target."""
        hl_obs = extract_hl_obs(full_obs, self.n_balls)
        valid = get_valid_mask(full_obs)

        if not np.any(valid):
            return 0  # all sorted, doesn't matter

        target = self.hl_policy.select_action(hl_obs, epsilon, valid)
        self.current_target = target
        self.steps_on_target = 0
        return target

    def select_action(self, full_obs: np.ndarray) -> np.ndarray:
        """Low-level selects motor action for current target ball."""
        if self.current_target is None:
            return np.zeros(4, dtype=np.float32)

        fake_obs = make_1ball_obs(full_obs, self.current_target)
        action = self.low_level.select_action(fake_obs, deterministic=True)
        self.steps_on_target += 1
        return action

    #  DQN Training 

    def update_hl(self, batch_size: int = 64) -> dict:
        """One DQN update step for the high-level policy."""
        if self.hl_buffer.size < batch_size:
            return {}

        obs, action, reward, next_obs, done, valid_masks = \
            self.hl_buffer.sample(batch_size)

        # Current Q-values
        q_values = self.hl_policy(obs)
        q_selected = q_values.gather(1, action.unsqueeze(1).long()).squeeze(1)

        # Target Q-values (with valid masking)
        with torch.no_grad():
            q_next = self.hl_target(next_obs)
            # Mask invalid actions in next state
            for i in range(batch_size):
                invalid = ~torch.from_numpy(valid_masks[i])
                q_next[i][invalid] = -1e6
            q_next_max = q_next.max(dim=1).values
            target = reward + self.gamma * (1 - done) * q_next_max

        loss = F.mse_loss(q_selected, target)

        self.hl_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.hl_policy.parameters(), 1.0)
        self.hl_optimizer.step()

        self.hl_updates += 1
        if self.hl_updates % self.target_update_freq == 0:
            self.hl_target.load_state_dict(self.hl_policy.state_dict())

        return {"hl_loss": loss.item(), "hl_q_mean": q_selected.mean().item()}

    def save(self, path):
        torch.save({
            "hl_policy": self.hl_policy.state_dict(),
            "hl_target": self.hl_target.state_dict(),
            "hl_optimizer": self.hl_optimizer.state_dict(),
            "hl_updates": self.hl_updates,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.hl_policy.load_state_dict(ckpt["hl_policy"])
        self.hl_target.load_state_dict(ckpt["hl_target"])
        self.hl_optimizer.load_state_dict(ckpt["hl_optimizer"])
        self.hl_updates = ckpt["hl_updates"]