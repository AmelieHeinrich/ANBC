"""Small per-block model for BC7 mode-6 prediction.

Input: one 4x4 RGBA8 block, flattened to 64 floats in [0,1].
Output: endpoint0 (4), endpoint1 (4), 16 per-pixel interpolation factors.
"""

from __future__ import annotations

import torch
import torch.nn as nn

BLOCK_INPUT_DIM = 4 * 4 * 4  # 16 pixels * 4 channels
OUTPUT_DIM = 4 + 4 + 16  # endpoint0 + endpoint1 + 16 interpolation factors


class BC7Mode6MLP(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_hidden_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(BLOCK_INPUT_DIM, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, OUTPUT_DIM))
        self.net = nn.Sequential(*layers)

    def forward(self, block: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """block: (N, 64) flattened RGBA block in [0,1].
        Returns (endpoint0, endpoint1, interp), shapes (N,4), (N,4), (N,16),
        all passed through sigmoid so they lie in [0,1]."""
        raw = self.net(block)
        endpoint0 = torch.sigmoid(raw[:, 0:4])
        endpoint1 = torch.sigmoid(raw[:, 4:8])
        interp = torch.sigmoid(raw[:, 8:24])
        return endpoint0, endpoint1, interp


MODE5_OUTPUT_DIM = 3 + 3 + 16 + 1 + 1 + 16  # rgb endpoints + rgb interp + alpha endpoints + alpha interp


class BC7Mode5MLP(nn.Module):
    """Predicts BC7 mode-5 params: RGB and alpha get independent endpoints
    and independent per-pixel interpolation factors, so alpha can vary
    without dragging color reconstruction down with it (mode 6's failure
    mode on real alpha-cutout textures)."""

    def __init__(self, hidden_dim: int = 128, num_hidden_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(BLOCK_INPUT_DIM, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, MODE5_OUTPUT_DIM))
        self.net = nn.Sequential(*layers)

    def forward(self, block: torch.Tensor):
        """block: (N, 64) flattened RGBA block in [0,1].
        Returns (endpoint0_rgb, endpoint1_rgb, interp_rgb,
                 endpoint0_a, endpoint1_a, interp_a)."""
        raw = self.net(block)
        endpoint0_rgb = torch.sigmoid(raw[:, 0:3])
        endpoint1_rgb = torch.sigmoid(raw[:, 3:6])
        interp_rgb = torch.sigmoid(raw[:, 6:22])
        endpoint0_a = torch.sigmoid(raw[:, 22:23])
        endpoint1_a = torch.sigmoid(raw[:, 23:24])
        interp_a = torch.sigmoid(raw[:, 24:40])
        return endpoint0_rgb, endpoint1_rgb, interp_rgb, endpoint0_a, endpoint1_a, interp_a


BC5_INPUT_DIM = 4 * 4 * 2  # 16 pixels * RG
BC5_OUTPUT_DIM = 1 + 1 + 16 + 1 + 1 + 16  # R endpoints + R interp + G endpoints + G interp


class BC5MLP(nn.Module):
    """Predicts BC5 params: R and G are two fully independent BC4 lines
    (own endpoints, own per-pixel interpolation factors), predicted by one
    net. Superseded -- the shipped BC5 encoder is network-free (see
    bc5_codec.py); kept for reference/experiments."""

    def __init__(self, hidden_dim: int = 128, num_hidden_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(BC5_INPUT_DIM, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, BC5_OUTPUT_DIM))
        self.net = nn.Sequential(*layers)

    def forward(self, block: torch.Tensor):
        """block: (N, 32) flattened RG block in [0,1].
        Returns (endpoint0_r, endpoint1_r, interp_r, endpoint0_g, endpoint1_g, interp_g),
        shapes (N,1), (N,1), (N,16) per channel, all sigmoid-ed into [0,1]."""
        raw = self.net(block)
        endpoint0_r = torch.sigmoid(raw[:, 0:1])
        endpoint1_r = torch.sigmoid(raw[:, 1:2])
        interp_r = torch.sigmoid(raw[:, 2:18])
        endpoint0_g = torch.sigmoid(raw[:, 18:19])
        endpoint1_g = torch.sigmoid(raw[:, 19:20])
        interp_g = torch.sigmoid(raw[:, 20:36])
        return endpoint0_r, endpoint1_r, interp_r, endpoint0_g, endpoint1_g, interp_g


class BC7Mode6CNN(nn.Module):
    """CNN variant operating directly on (N, 4, 4, 4) blocks (channels-first)."""

    def __init__(self, hidden_channels: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_channels * 4 * 4, OUTPUT_DIM)

    def forward(self, block_nchw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """block_nchw: (N, 4, 4, 4) RGBA-channels-first block in [0,1]."""
        feat = self.conv(block_nchw)
        raw = self.head(feat.flatten(1))
        endpoint0 = torch.sigmoid(raw[:, 0:4])
        endpoint1 = torch.sigmoid(raw[:, 4:8])
        interp = torch.sigmoid(raw[:, 8:24])
        return endpoint0, endpoint1, interp
