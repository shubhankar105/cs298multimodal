"""Multi-task loss for MERA training.

Components
----------
1. **Primary loss**: Weighted cross-entropy on final fused prediction.
2. **Auxiliary loss A**: Cross-entropy on Pipeline A standalone prediction.
3. **Auxiliary loss B**: Cross-entropy on Pipeline B standalone prediction.
4. **Consistency loss**: KL divergence between the fused prediction and each
   single-modality prediction (detached).  This encourages modality-dropout
   robustness.

.. math::

    \\mathcal{L} = \\lambda_1 L_{\\text{primary}}
                  + \\lambda_2 L_{\\text{aux\\_text}}
                  + \\lambda_3 L_{\\text{aux\\_audio}}
                  + \\lambda_4 L_{\\text{consistency}}

Default weights: λ₁ = 1.0, λ₂ = 0.3, λ₃ = 0.3, λ₄ = 0.2.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MERALoss(nn.Module):
    """Multi-task loss combining primary, auxiliary, and consistency terms."""

    def __init__(
        self,
        num_classes: int = 4,
        class_weights: Optional[torch.Tensor] = None,
        lambda_primary: float = 1.0,
        lambda_aux_text: float = 0.3,
        lambda_aux_audio: float = 0.3,
        lambda_consistency: float = 0.2,
        label_smoothing: float = 0.1,
    ):
        super().__init__()

        self.lambda_primary = lambda_primary
        self.lambda_aux_text = lambda_aux_text
        self.lambda_aux_audio = lambda_aux_audio
        self.lambda_consistency = lambda_consistency

        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute all loss components.

        Args:
            outputs: Dict from ``MERAModel.forward()`` with keys
                ``final_logits``, ``text_logits``, ``audio_logits``.
            targets: ``(batch,)`` ground-truth emotion indices.

        Returns:
            Dict with ``total``, ``primary``, ``aux_text``, ``aux_audio``,
            ``consistency`` — all scalar tensors.
        """
        losses: dict[str, torch.Tensor] = {}

        # Primary loss on fused prediction
        losses["primary"] = self.ce_loss(outputs["final_logits"], targets)

        # Auxiliary losses on each pipeline's standalone prediction
        losses["aux_text"] = self.ce_loss(outputs["text_logits"], targets)
        losses["aux_audio"] = self.ce_loss(outputs["audio_logits"], targets)

        # Consistency loss: KL(final ‖ text) + KL(final ‖ audio)
        final_log_probs = F.log_softmax(outputs["final_logits"], dim=1)
        text_probs = F.softmax(outputs["text_logits"].detach(), dim=1)
        audio_probs = F.softmax(outputs["audio_logits"].detach(), dim=1)

        kl_text = F.kl_div(final_log_probs, text_probs, reduction="batchmean")
        kl_audio = F.kl_div(final_log_probs, audio_probs, reduction="batchmean")
        losses["consistency"] = (kl_text + kl_audio) / 2.0

        # Weighted total
        losses["total"] = (
            self.lambda_primary * losses["primary"]
            + self.lambda_aux_text * losses["aux_text"]
            + self.lambda_aux_audio * losses["aux_audio"]
            + self.lambda_consistency * losses["consistency"]
        )

        return losses


class PipelineLoss(nn.Module):
    """Simple cross-entropy loss for standalone pipeline training (A or B)."""

    def __init__(
        self,
        num_classes: int = 4,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            logits: ``(batch, num_classes)``
            targets: ``(batch,)``

        Returns:
            Dict with ``total`` and ``ce`` loss.
        """
        ce = self.ce_loss(logits, targets)
        return {"total": ce, "ce": ce}


def compute_class_weights(
    labels: list[int] | torch.Tensor,
    num_classes: int = 4,
    method: str = "inverse_freq",
) -> torch.Tensor:
    """Compute class weights for imbalanced datasets.

    Args:
        labels: 1-D integer label list or tensor.
        num_classes: Total number of classes.
        method: ``"inverse_freq"`` or ``"effective_num"`` (from Cui et al. 2019).

    Returns:
        ``(num_classes,)`` float32 weight tensor (sum-normalised).
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()

    counts = torch.zeros(num_classes)
    for lbl in labels:
        counts[lbl] += 1

    if method == "inverse_freq":
        weights = 1.0 / (counts + 1e-6)
    elif method == "effective_num":
        beta = 0.999
        effective = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / (effective + 1e-6)
    else:
        raise ValueError(f"Unknown method: {method}")

    weights = weights / weights.sum() * num_classes
    return weights.float()
