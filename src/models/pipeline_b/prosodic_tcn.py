"""Stream 2: Temporal Convolutional Network for Prosodic Contour Modelling.

***** THIS IS A KEY NOVEL CONTRIBUTION *****

Instead of extracting summary statistics from prosodic features (mean F0,
std energy, etc.) — which is what the *entire* SER field does (openSMILE →
SVM/MLP) — we treat prosodic contours as continuous 1-D signals and learn
emotion-discriminative temporal patterns directly.

The TCN uses dilated **causal** convolutions to capture multi-scale dynamics:

* **Small dilation (1–2)**: local pitch perturbations (jitter, rapid F0 changes)
* **Medium dilation (4–8)**: phrase-level intonation (question rises, anger peaks)
* **Large dilation (16–32)**: utterance-level contour shapes (sadness = gradual fall)

Architecture
------------
1. Channel attention → learns which prosodic features matter per input.
2. 1 × 1 input projection → ``tcn_channels``.
3. 6 residual TCN blocks with dilations ``[1, 2, 4, 8, 16, 32]``.
4. Temporal attention pooling → fixed-size embedding.

Receptive field
    Each block adds ``(kernel_size − 1) × dilation`` frames.
    Total = 4 × (1+2+4+8+16+32) = **252 frames = 2.52 s** at 10 ms/frame.

Input:  ``(batch, 10, T)``  — 10-channel prosodic contours.
Output: ``(batch, output_dim)``  — default 128-d embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class TemporalBlock(nn.Module):
    """Single residual block with dilated causal convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Causal padding: left-pad so output length == input length
        self.causal_padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.causal_padding, dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=self.causal_padding, dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        # 1 × 1 residual projection when channel count changes
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(batch, channels, T)``
        Returns:
            ``(batch, out_channels, T)``
        """
        T = x.shape[2]
        residual = self.residual(x)

        out = self.conv1(x)[:, :, :T]     # causal trim
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)[:, :, :T]    # causal trim
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        return self.relu(out + residual)


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation–style attention over prosodic channels.

    Learns which prosodic features (F0, energy, jitter, …) are most
    informative for a given input.
    """

    def __init__(self, num_channels: int, reduction: int = 4):
        super().__init__()
        mid = max(num_channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, num_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(batch, channels, T)``
        Returns:
            ``(batch, channels, T)`` — channel-reweighted.
        """
        # Global average pooling over time
        gap = x.mean(dim=2)                       # (B, C)
        weights = self.fc(gap).unsqueeze(2)       # (B, C, 1)
        return x * weights


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------


class ProsodicTCN(nn.Module):
    """Temporal Convolutional Network for prosodic contour modelling."""

    def __init__(
        self,
        in_channels: int = 10,
        tcn_channels: int = 64,
        num_blocks: int = 6,
        kernel_size: int = 5,
        output_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Channel attention on raw prosodic features
        self.channel_attention = ChannelAttention(in_channels)

        # 1 × 1 input projection
        self.input_proj = nn.Conv1d(in_channels, tcn_channels, 1)

        # Residual TCN blocks with exponentially increasing dilation
        self.blocks = nn.ModuleList([
            TemporalBlock(
                tcn_channels, tcn_channels,
                kernel_size, dilation=2 ** i, dropout=dropout,
            )
            for i in range(num_blocks)
        ])

        # Temporal attention pooling
        self.temporal_attention = nn.Sequential(
            nn.Linear(tcn_channels, tcn_channels // 2),
            nn.Tanh(),
            nn.Linear(tcn_channels // 2, 1),
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(tcn_channels, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        prosody: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            prosody: ``(batch, 10, T)`` — 10-channel prosodic contours.
            mask:    ``(batch, T)``     — True for valid frames.

        Returns:
            ``(batch, output_dim)`` embedding.
        """
        # Channel attention
        x = self.channel_attention(prosody)   # (B, 10, T)

        # Project to hidden dim
        x = self.input_proj(x)                # (B, C, T)

        # TCN blocks
        for block in self.blocks:
            x = block(x)                      # (B, C, T)

        # Attention pooling
        x_t = x.permute(0, 2, 1)                                  # (B, T, C)
        attn_scores = self.temporal_attention(x_t).squeeze(-1)     # (B, T)

        if mask is not None:
            T = attn_scores.shape[1]
            m = mask[:, :T] if mask.shape[1] >= T else nn.functional.pad(
                mask, (0, T - mask.shape[1]), value=False,
            )
            attn_scores = attn_scores.masked_fill(~m, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(1)  # (B, 1, T)
        pooled = torch.bmm(attn_weights, x_t).squeeze(1)              # (B, C)

        return self.output_proj(pooled)                                # (B, D)
