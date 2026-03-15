"""Gated fusion with linguistic confidence signal.

**Novel contribution**: The gate learns a per-sample modality weighting
that is explicitly informed by DeBERTa's attention entropy.

* High entropy → text pipeline is uncertain → gate shifts toward audio.
* Low entropy  → text pipeline is confident → gate relies more on text.

The gate network takes ``[text_repr, audio_repr, projected_entropy]`` as
input and outputs a 2-element sigmoid vector, normalised to sum to 1.

Input:
    text_repr         ``(B, D)``
    audio_repr        ``(B, D)``
    attention_entropy ``(B, num_layers)``   (e.g. 12 for DeBERTa-base)

Output:
    fused         ``(B, D)``
    gate_weights  ``(B, 2)``  — for interpretability / visualisation
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """Entropy-informed gated modality fusion."""

    def __init__(
        self,
        repr_dim: int = 256,
        entropy_dim: int = 12,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Project attention entropy to a small feature vector
        self.entropy_proj = nn.Sequential(
            nn.Linear(entropy_dim, 32),
            nn.ReLU(),
        )

        # Gate: [text_repr ‖ audio_repr ‖ entropy_feat] → 2 weights
        gate_input_dim = repr_dim * 2 + 32
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, repr_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(repr_dim, 2),
            nn.Sigmoid(),
        )

    def forward(
        self,
        text_repr: torch.Tensor,
        audio_repr: torch.Tensor,
        attention_entropy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text_repr:         ``(B, D)``
            audio_repr:        ``(B, D)``
            attention_entropy: ``(B, entropy_dim)``

        Returns:
            fused:        ``(B, D)``
            gate_weights: ``(B, 2)``  — [text_weight, audio_weight]
        """
        entropy_feat = self.entropy_proj(attention_entropy)           # (B, 32)

        gate_input = torch.cat([text_repr, audio_repr, entropy_feat], dim=1)
        gate_raw = self.gate(gate_input)                              # (B, 2)

        # Normalise to sum-to-one
        gate_weights = gate_raw / (gate_raw.sum(dim=1, keepdim=True) + 1e-8)

        # Weighted combination
        fused = gate_weights[:, 0:1] * text_repr + gate_weights[:, 1:2] * audio_repr

        return fused, gate_weights
