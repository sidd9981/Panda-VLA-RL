"""
CNN Encoder for vision-based SAC.

Dual camera input:
    - Overhead cam (64x64x3): global scene view — where are balls, bins
    - Wrist cam (64x64x3): close-up of gripper — grasping precision

Architecture:
    Each image → small CNN → 128D features
    Concat both → 256D visual features
    Concat with proprio (7D: ee_pos, ee_vel, finger_width) → 263D
    → SAC actor/critic

Design choices:
    - Small CNN (3 conv layers, ~50K params per stream) — fast on M3 Pro
    - No frame stacking — single frame is enough for this task
    - LayerNorm on features (matches our SAC architecture)
    - Shared encoder between actor and critic (standard in DrQ)
"""

import torch
import torch.nn as nn
import numpy as np


class SingleCameraEncoder(nn.Module):
    """
    Small CNN: 64x64x3 → 128D feature vector.
    
    3 conv layers with stride 2 → 8x8 feature map → flatten → linear → 128D
    ~50K params. Fast enough for real-time on M3 Pro.
    """
    
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 64→32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32→16
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # 16→8
            nn.ReLU(),
        )
        # 64 channels × 8 × 8 = 4096
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: (batch, 3, 64, 64) float32 in [0, 1]
        returns: (batch, out_dim)
        """
        h = self.conv(img)
        h = h.reshape(h.size(0), -1)
        return self.fc(h)


class DualCameraEncoder(nn.Module):
    """
    Dual camera encoder: overhead + wrist → 256D visual features.
    
    Each camera gets its own CNN (no weight sharing — they see very
    different things). Features are concatenated.
    
    Total visual feature dim: 128 + 128 = 256
    With proprio (7D): 256 + 7 = 263D input to SAC
    """
    
    def __init__(self, per_cam_dim: int = 128):
        super().__init__()
        self.overhead_enc = SingleCameraEncoder(per_cam_dim)
        self.wrist_enc = SingleCameraEncoder(per_cam_dim)
        self.out_dim = per_cam_dim * 2  # 256
    
    def forward(self, overhead: torch.Tensor, wrist: torch.Tensor) -> torch.Tensor:
        """
        overhead: (batch, 3, 64, 64)
        wrist:    (batch, 3, 64, 64)
        returns:  (batch, 256)
        """
        oh_feat = self.overhead_enc(overhead)
        wr_feat = self.wrist_enc(wrist)
        return torch.cat([oh_feat, wr_feat], dim=-1)


class VisionEncoder(nn.Module):
    """
    Full vision encoder: dual cameras + proprio → feature vector for SAC.
    
    This is the drop-in replacement for the state obs.
    State SAC: 51D → actor/critic
    Vision SAC: (overhead_img, wrist_img, proprio_7D) → 263D → actor/critic
    """
    
    def __init__(self, per_cam_dim: int = 128, proprio_dim: int = 7):
        super().__init__()
        self.camera_enc = DualCameraEncoder(per_cam_dim)
        self.proprio_dim = proprio_dim
        self.out_dim = self.camera_enc.out_dim + proprio_dim  # 263
    
    def forward(self, overhead: torch.Tensor, wrist: torch.Tensor,
                proprio: torch.Tensor) -> torch.Tensor:
        """
        overhead: (batch, 3, 64, 64)
        wrist:    (batch, 3, 64, 64)
        proprio:  (batch, 7) — ee_pos(3) + ee_vel(3) + finger_width(1)
        returns:  (batch, 263)
        """
        vis_feat = self.camera_enc(overhead, wrist)
        return torch.cat([vis_feat, proprio], dim=-1)


def preprocess_image(img: np.ndarray, target_size: int = 64) -> np.ndarray:
    """
    Convert raw MuJoCo render (H, W, 3) uint8 → (3, H, W) float32 [0, 1].
    Resizes to target_size if needed.
    """
    import cv2
    if img.shape[0] != target_size or img.shape[1] != target_size:
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    # HWC → CHW, uint8 → float32 [0, 1]
    return img.transpose(2, 0, 1).astype(np.float32) / 255.0


def random_crop(img: torch.Tensor, out_size: int = 64) -> torch.Tensor:
    """
    Random crop for data augmentation (DrQ-style).
    Input: (batch, 3, H, W) where H, W >= out_size
    Output: (batch, 3, out_size, out_size)
    """
    b, c, h, w = img.shape
    assert h >= out_size and w >= out_size
    if h == out_size and w == out_size:
        return img
    crop_h = torch.randint(0, h - out_size + 1, (b,))
    crop_w = torch.randint(0, w - out_size + 1, (b,))
    cropped = torch.zeros(b, c, out_size, out_size, device=img.device, dtype=img.dtype)
    for i in range(b):
        cropped[i] = img[i, :, crop_h[i]:crop_h[i]+out_size, crop_w[i]:crop_w[i]+out_size]
    return cropped


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("CNN Encoder — Smoke Test")
    print("=" * 40)
    
    enc = VisionEncoder(per_cam_dim=128, proprio_dim=7)
    total_params = sum(p.numel() for p in enc.parameters())
    print(f"  Total params: {total_params:,}")
    print(f"  Output dim:   {enc.out_dim}")
    
    # Test forward pass
    batch = 4
    overhead = torch.randn(batch, 3, 64, 64)
    wrist = torch.randn(batch, 3, 64, 64)
    proprio = torch.randn(batch, 7)
    
    feat = enc(overhead, wrist, proprio)
    print(f"  Input:  overhead={overhead.shape}, wrist={wrist.shape}, proprio={proprio.shape}")
    print(f"  Output: {feat.shape}")
    assert feat.shape == (batch, 263)
    
    # Test preprocess
    fake_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    processed = preprocess_image(fake_img)
    print(f"  Preprocess: {fake_img.shape} → {processed.shape}, range=[{processed.min():.2f}, {processed.max():.2f}]")
    assert processed.shape == (3, 64, 64)
    assert processed.dtype == np.float32
    
    print("\n  All tests passed!")