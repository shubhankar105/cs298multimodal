"""Pipeline A: DeBERTa-v3-base text emotion encoder.

Architecture
------------
1. DeBERTa-v3-base encodes transcript text → ``(batch, seq_len, 768)``
2. ``[CLS]`` token embedding is used for classification and fusion.
3. Attention entropy is computed across all 12 layers as a confidence
   signal for the downstream fusion module (novel contribution).
4. Classification head: ``Linear(768→256) → ReLU → Dropout → Linear(256→4)``

Fine-tuning strategy
--------------------
1. Pre-train on GoEmotions (mapped to 4 classes) for 3 epochs.
2. Domain-adapt on IEMOCAP transcripts for 15 epochs.
3. Learning rate 2 × 10⁻⁵ with linear warmup.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from src.models.pipeline_a.attention_entropy import AttentionEntropyModule

logger = logging.getLogger(__name__)


class TextEmotionEncoder(nn.Module):
    """DeBERTa-v3-base with attention entropy extraction.

    Returns three outputs on every forward pass:

    * **logits** ``(batch, num_classes)`` – emotion predictions
    * **representation** ``(batch, hidden_dim)`` – 256-d vector for fusion
    * **attention_entropy** ``(batch, num_layers)`` – per-layer confidence
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

        from transformers import DebertaV2Model, DebertaV2Tokenizer

        self.deberta = DebertaV2Model.from_pretrained(
            model_name,
            output_attentions=True,
        )
        self._tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)

        self.hidden_size: int = self.deberta.config.hidden_size        # 768
        self.num_layers: int = self.deberta.config.num_hidden_layers   # 12
        self.num_heads: int = self.deberta.config.num_attention_heads  # 12

        # Optionally freeze bottom N layers
        if freeze_layers > 0:
            for i, layer in enumerate(self.deberta.encoder.layer):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Projection for fusion representation
        self.representation_proj = nn.Linear(self.hidden_size, hidden_dim)

        # Attention entropy module
        self.entropy_module = AttentionEntropyModule()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids:      ``(batch, seq_len)`` token IDs.
            attention_mask: ``(batch, seq_len)`` mask (1 = real, 0 = pad).

        Returns:
            logits:             ``(batch, num_classes)``
            representation:     ``(batch, hidden_dim)``
            attention_entropy:  ``(batch, num_layers)``
        """
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )

        # [CLS] token is at position 0
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (B, 768)

        logits = self.classifier(cls_embedding)             # (B, 4)
        representation = self.representation_proj(cls_embedding)  # (B, 256)
        attention_entropy = self.entropy_module(outputs.attentions)  # (B, 12)

        return logits, representation, attention_entropy

    # ------------------------------------------------------------------
    # Tokenization helper
    # ------------------------------------------------------------------

    def tokenize(
        self,
        texts: list[str],
        max_length: int = 128,
        device: Optional[torch.device] = None,
    ) -> dict[str, torch.Tensor]:
        """Tokenize a list of transcript strings for this encoder.

        Args:
            texts: List of raw text strings.
            max_length: Maximum token length (truncated if exceeded).
            device: Optional device to move tensors to.

        Returns:
            Dict with ``input_ids`` and ``attention_mask`` tensors.
        """
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if device is not None:
            encoded = {k: v.to(device) for k, v in encoded.items()}
        return encoded
