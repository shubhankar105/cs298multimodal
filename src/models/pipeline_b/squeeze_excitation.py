"""Squeeze-and-Excitation sub-stream fusion.

Dynamically re-weights the three paralinguistic sub-streams per sample:

* Whispered anger → prosody stream up-weighted (clear prosodic pattern)
* Noisy audio     → HuBERT stream up-weighted (robust pretrained representations)
* Clear speech    → CNN-BiLSTM may dominate (rich spectral detail)

Input:  Concatenated sub-stream embeddings ``(batch, 256 + 128 + 256 = 640)``
Output: Fused audio embedding ``(batch, output_dim)``
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SubStreamFusion(nn.Module):
    """SE-based fusion for three audio sub-streams."""

    def __init__(
        self,
        stream1_dim: int = 256,
        stream2_dim: int = 128,
        stream3_dim: int = 256,
        output_dim: int = 256,
        reduction: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        total_dim = stream1_dim + stream2_dim + stream3_dim

        # SE block: squeeze → excite → reweight
        self.se = nn.Sequential(
            nn.Linear(total_dim, total_dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(total_dim // reduction, total_dim),
            nn.Sigmoid(),
        )

        # Output projection
        self.projection = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        stream1_emb: torch.Tensor,
        stream2_emb: torch.Tensor,
        stream3_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            stream1_emb: ``(batch, 256)`` — CNN-BiLSTM
            stream2_emb: ``(batch, 128)`` — Prosodic TCN
            stream3_emb: ``(batch, 256)`` — HuBERT head

        Returns:
            ``(batch, output_dim)`` fused embedding.
        """
        concat = torch.cat([stream1_emb, stream2_emb, stream3_emb], dim=1)
        weights = self.se(concat)
        reweighted = concat * weights
        return self.projection(reweighted)
