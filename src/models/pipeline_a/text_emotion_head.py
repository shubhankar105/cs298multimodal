"""Standalone classification head for Pipeline A evaluation.

Used when evaluating Pipeline A in isolation (ablation: "Text-Only").
Wraps the TextEmotionEncoder and exposes a simple
``predict(texts) → emotion_labels`` interface.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.models.pipeline_a.text_encoder import TextEmotionEncoder


class TextEmotionHead(nn.Module):
    """Thin wrapper that runs TextEmotionEncoder and returns only logits.

    This is used for standalone Pipeline-A training and evaluation so that
    training loops can treat it like a standard ``(input → logits)`` model.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_classes: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        freeze_layers: int = 0,
    ):
        super().__init__()
        self.encoder = TextEmotionEncoder(
            model_name=model_name,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
            freeze_layers=freeze_layers,
        )
        self.num_classes = num_classes

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return emotion logits only (drop representation & entropy).

        Args:
            input_ids:      ``(batch, seq_len)``
            attention_mask: ``(batch, seq_len)``

        Returns:
            ``(batch, num_classes)`` logits.
        """
        logits, _, _ = self.encoder(input_ids, attention_mask)
        return logits

    def tokenize(self, texts: list[str], max_length: int = 128, device: Optional[torch.device] = None):
        """Delegate to the underlying encoder's tokenizer."""
        return self.encoder.tokenize(texts, max_length=max_length, device=device)

    @torch.no_grad()
    def predict(self, texts: list[str], device: Optional[torch.device] = None) -> list[int]:
        """Convenience: tokenize → forward → argmax.

        Args:
            texts: Raw transcript strings.
            device: Target device.

        Returns:
            List of predicted emotion indices (0–3).
        """
        self.eval()
        encoded = self.tokenize(texts, device=device)
        logits = self.forward(encoded["input_ids"], encoded["attention_mask"])
        return logits.argmax(dim=-1).tolist()
