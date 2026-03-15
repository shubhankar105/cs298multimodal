"""Bidirectional cross-modal attention between text and audio representations.

**Novel contribution**: learns complementary interactions between modalities.

* Text queries attend to audio keys/values → "how the text *sounds*"
* Audio queries attend to text keys/values → "what the voice is *saying*"

Example: the word "fine" is neutral in text, but sarcastic when spoken
with a specific pitch pattern.  Cross-modal attention lets the text
representation absorb prosodic cues and vice versa.

Input:  text_repr ``(B, D)`` + audio_repr ``(B, D)``
Output: text_enhanced ``(B, D)`` + audio_enhanced ``(B, D)``
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):
    """Bidirectional cross-modal attention with residual connections."""

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Text → Audio attention (text queries, audio keys/values)
        self.text_to_audio = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )

        # Audio → Text attention (audio queries, text keys/values)
        self.audio_to_text = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )

        self.norm_t2a = nn.LayerNorm(dim)
        self.norm_a2t = nn.LayerNorm(dim)

    def forward(
        self,
        text_repr: torch.Tensor,
        audio_repr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text_repr:  ``(batch, dim)``
            audio_repr: ``(batch, dim)``

        Returns:
            text_enhanced:  ``(batch, dim)``
            audio_enhanced: ``(batch, dim)``
        """
        # Reshape to sequence format: (B, 1, D) — single "token" per modality
        t = text_repr.unsqueeze(1)
        a = audio_repr.unsqueeze(1)

        # Cross-modal attention
        t2a, _ = self.text_to_audio(query=t, key=a, value=a)
        a2t, _ = self.audio_to_text(query=a, key=t, value=t)

        # Residual + LayerNorm → (B, D)
        text_enhanced = self.norm_t2a(t + t2a).squeeze(1)
        audio_enhanced = self.norm_a2t(a + a2t).squeeze(1)

        return text_enhanced, audio_enhanced
