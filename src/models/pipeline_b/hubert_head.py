"""Stream 3: Learnable weighted sum over frozen HuBERT layers + attention pooling.

HuBERT-Large has 25 layers (1 CNN input + 24 transformer) with different
properties:

* Early layers  → raw acoustic features (spectral, phonetic)
* Middle layers → phoneme-level representations
* Late layers   → higher-level linguistic features

Instead of using only the last layer (common but suboptimal), we learn a
softmax-normalised weighted combination across **all** layers, akin to
ELMo's scalar-mix approach for text.

Input:  ``(batch, 25, T, 1024)`` — pre-cached HuBERT hidden states.
Output: ``(batch, output_dim)``  — default 256-d embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HuBERTWeightedHead(nn.Module):
    """ELMo-style weighted-sum head over HuBERT hidden layers."""

    def __init__(
        self,
        num_layers: int = 25,
        hidden_dim: int = 1024,
        output_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Learnable layer weights (initialised uniform)
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

        # Scalar scaling factor (ELMo γ)
        self.scalar = nn.Parameter(torch.ones(1))

        # 1024 → output_dim projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Attention pooling over time frames
        self.attention = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.Tanh(),
            nn.Linear(output_dim // 2, 1),
        )

    def forward(
        self,
        hubert_embeddings: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hubert_embeddings: ``(batch, 25, T, 1024)`` cached HuBERT outputs.
            mask: ``(batch, T)`` boolean mask (True = valid).

        Returns:
            ``(batch, output_dim)`` embedding.
        """
        # Softmax-normalise layer weights
        normed_weights = torch.softmax(self.layer_weights, dim=0)  # (L,)

        # Weighted sum across layers → (B, T, D)
        weighted = torch.einsum("l, bltd -> btd", normed_weights, hubert_embeddings)
        weighted = weighted * self.scalar

        # Project → (B, T, output_dim)
        projected = self.projection(weighted)

        # Attention pooling → (B, output_dim)
        attn_scores = self.attention(projected).squeeze(-1)  # (B, T)

        if mask is not None:
            T = attn_scores.shape[1]
            m = mask[:, :T] if mask.shape[1] >= T else nn.functional.pad(
                mask, (0, T - mask.shape[1]), value=False,
            )
            attn_scores = attn_scores.masked_fill(~m, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(1)  # (B, 1, T)
        pooled = torch.bmm(attn_weights, projected).squeeze(1)         # (B, D)

        return pooled
