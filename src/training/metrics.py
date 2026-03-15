"""Evaluation metrics for speech emotion recognition.

Standard metrics reported in IEMOCAP literature:
- **Weighted Accuracy (WA)**: standard accuracy.
- **Unweighted Accuracy (UA)**: mean of per-class accuracies (handles imbalance).
- **Per-class F1 score**.
- **Confusion matrix**.

``MetricTracker`` accumulates predictions across batches and computes
epoch-level metrics in one call.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import torch

EMOTION_LABELS = ["angry", "happy", "sad", "neutral"]


class MetricTracker:
    """Accumulates predictions and targets across batches.

    Usage::

        tracker = MetricTracker(num_classes=4)
        for batch in dataloader:
            ...
            tracker.update(logits, targets)
        results = tracker.compute()
        tracker.reset()
    """

    def __init__(self, num_classes: int = 4, label_names: Optional[list[str]] = None):
        self.num_classes = num_classes
        self.label_names = label_names or EMOTION_LABELS[:num_classes]
        self.reset()

    def reset(self) -> None:
        """Clear accumulated predictions."""
        self._preds: list[int] = []
        self._targets: list[int] = []

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Add a batch of predictions.

        Args:
            logits: ``(batch, num_classes)`` raw model output.
            targets: ``(batch,)`` integer labels.
        """
        preds = logits.argmax(dim=1).detach().cpu().tolist()
        tgts = targets.detach().cpu().tolist()
        self._preds.extend(preds)
        self._targets.extend(tgts)

    def compute(self) -> dict:
        """Compute all metrics from accumulated predictions.

        Returns:
            Dict with keys: ``wa``, ``ua``, ``per_class_f1``, ``macro_f1``,
            ``confusion_matrix``, ``per_class_accuracy``, ``support``.
        """
        preds = np.array(self._preds)
        targets = np.array(self._targets)

        if len(preds) == 0:
            return self._empty_results()

        wa = compute_weighted_accuracy(preds, targets)
        ua, per_class_acc = compute_unweighted_accuracy(preds, targets, self.num_classes)
        per_class_f1, macro_f1 = compute_f1_scores(preds, targets, self.num_classes)
        cm = compute_confusion_matrix(preds, targets, self.num_classes)

        support = {}
        for c in range(self.num_classes):
            name = self.label_names[c] if c < len(self.label_names) else str(c)
            support[name] = int((targets == c).sum())

        return {
            "wa": wa,
            "ua": ua,
            "macro_f1": macro_f1,
            "per_class_f1": {
                self.label_names[c]: per_class_f1[c]
                for c in range(self.num_classes)
            },
            "per_class_accuracy": {
                self.label_names[c]: per_class_acc[c]
                for c in range(self.num_classes)
            },
            "confusion_matrix": cm,
            "support": support,
            "total_samples": len(preds),
        }

    def _empty_results(self) -> dict:
        return {
            "wa": 0.0,
            "ua": 0.0,
            "macro_f1": 0.0,
            "per_class_f1": {},
            "per_class_accuracy": {},
            "confusion_matrix": np.zeros((self.num_classes, self.num_classes), dtype=int),
            "support": {},
            "total_samples": 0,
        }


# ---------------------------------------------------------------------------
# Standalone metric functions
# ---------------------------------------------------------------------------

def compute_weighted_accuracy(
    preds: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Standard accuracy (correct / total)."""
    return float((preds == targets).mean())


def compute_unweighted_accuracy(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 4,
) -> tuple[float, dict[int, float]]:
    """Mean of per-class accuracies (handles class imbalance).

    Returns:
        Tuple of (UA scalar, dict mapping class_idx → per-class accuracy).
    """
    per_class: dict[int, float] = {}
    for c in range(num_classes):
        mask = targets == c
        if mask.sum() == 0:
            per_class[c] = 0.0
        else:
            per_class[c] = float((preds[mask] == c).mean())
    ua = float(np.mean(list(per_class.values())))
    return ua, per_class


def compute_f1_scores(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 4,
) -> tuple[dict[int, float], float]:
    """Per-class and macro-average F1 scores.

    Returns:
        Tuple of (per-class F1 dict, macro-average F1 float).
    """
    per_class_f1: dict[int, float] = {}

    for c in range(num_classes):
        tp = int(((preds == c) & (targets == c)).sum())
        fp = int(((preds == c) & (targets != c)).sum())
        fn = int(((preds != c) & (targets == c)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_class_f1[c] = f1

    macro_f1 = float(np.mean(list(per_class_f1.values())))
    return per_class_f1, macro_f1


def compute_confusion_matrix(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 4,
) -> np.ndarray:
    """Confusion matrix: ``cm[true][pred]``.

    Returns:
        ``(num_classes, num_classes)`` integer array.
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(targets, preds):
        cm[t, p] += 1
    return cm


def format_metrics(metrics: dict, label_names: Optional[list[str]] = None) -> str:
    """Pretty-print metrics for logging.

    Args:
        metrics: Dict returned by ``MetricTracker.compute()``.
        label_names: Optional label names for display.

    Returns:
        Multi-line formatted string.
    """
    lines = [
        f"WA: {metrics['wa']:.4f}  |  UA: {metrics['ua']:.4f}  |  Macro-F1: {metrics['macro_f1']:.4f}",
    ]
    if metrics.get("per_class_f1"):
        f1_parts = [f"{k}: {v:.3f}" for k, v in metrics["per_class_f1"].items()]
        lines.append(f"Per-class F1: {', '.join(f1_parts)}")
    if metrics.get("support"):
        sup_parts = [f"{k}: {v}" for k, v in metrics["support"].items()]
        lines.append(f"Support: {', '.join(sup_parts)}")
    return "\n".join(lines)
