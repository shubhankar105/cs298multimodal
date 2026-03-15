"""Full evaluation pipeline for the MERA system.

Supports:
- Loading a trained MERA, Pipeline A, or Pipeline B checkpoint.
- Running inference on a test set and computing all metrics
  (WA, UA, per-class F1, confusion matrix).
- 5-fold leave-one-session-out cross-validation on IEMOCAP with
  aggregate results (mean +/- std across folds).
- Saving structured results to JSON.
- Generating formatted LaTeX / Markdown results tables for the thesis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.metrics import (
    MetricTracker,
    EMOTION_LABELS,
    compute_confusion_matrix,
)
from src.data.collate import EMOTION_TO_IDX, IDX_TO_EMOTION

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    """Evaluation results for a single fold."""

    fold: int
    wa: float = 0.0
    ua: float = 0.0
    macro_f1: float = 0.0
    per_class_f1: dict[str, float] = field(default_factory=dict)
    per_class_accuracy: dict[str, float] = field(default_factory=dict)
    confusion_matrix: list[list[int]] = field(default_factory=list)
    support: dict[str, int] = field(default_factory=dict)
    total_samples: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AggregateResult:
    """Aggregated results across multiple folds (mean +/- std)."""

    experiment_name: str = ""
    num_folds: int = 0
    wa_mean: float = 0.0
    wa_std: float = 0.0
    ua_mean: float = 0.0
    ua_std: float = 0.0
    macro_f1_mean: float = 0.0
    macro_f1_std: float = 0.0
    per_class_f1_mean: dict[str, float] = field(default_factory=dict)
    per_class_f1_std: dict[str, float] = field(default_factory=dict)
    avg_confusion_matrix: list[list[float]] = field(default_factory=list)
    fold_results: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """Runs inference and computes metrics for any MERA model variant.

    Args:
        model: The model (MERAModel, TextEmotionEncoder, or AudioEmotionHead).
        device: Torch device.
        forward_fn: Callable ``(model, batch, device) -> (loss_dict, logits, targets)``.
            Same interface as the Trainer forward functions.
        num_classes: Number of emotion classes (default 4).
        label_names: Names for each class index.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        forward_fn,
        num_classes: int = 4,
        label_names: Optional[list[str]] = None,
    ):
        self.model = model
        self.device = device
        self.forward_fn = forward_fn
        self.num_classes = num_classes
        self.label_names = label_names or EMOTION_LABELS[:num_classes]

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> FoldResult:
        """Run inference on a dataloader and compute all metrics.

        Args:
            dataloader: Test/validation DataLoader.

        Returns:
            FoldResult with all computed metrics.
        """
        self.model.eval()
        tracker = MetricTracker(
            num_classes=self.num_classes,
            label_names=self.label_names,
        )

        all_preds = []
        all_targets = []

        for batch in dataloader:
            loss_dict, logits, targets = self.forward_fn(
                self.model, batch, self.device,
            )
            tracker.update(logits, targets)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

        metrics = tracker.compute()
        cm = metrics.get("confusion_matrix", np.zeros((self.num_classes, self.num_classes)))

        return FoldResult(
            fold=0,
            wa=metrics["wa"],
            ua=metrics["ua"],
            macro_f1=metrics["macro_f1"],
            per_class_f1=metrics.get("per_class_f1", {}),
            per_class_accuracy=metrics.get("per_class_accuracy", {}),
            confusion_matrix=cm.tolist() if isinstance(cm, np.ndarray) else cm,
            support=metrics.get("support", {}),
            total_samples=metrics.get("total_samples", 0),
        )


# ---------------------------------------------------------------------------
# Multi-fold evaluation
# ---------------------------------------------------------------------------

def aggregate_fold_results(
    fold_results: list[FoldResult],
    experiment_name: str = "",
) -> AggregateResult:
    """Aggregate metrics across folds: compute mean +/- std.

    Args:
        fold_results: List of FoldResult objects, one per fold.
        experiment_name: Name for this experiment (e.g. "MERA-Full").

    Returns:
        AggregateResult with mean/std statistics.
    """
    n = len(fold_results)
    if n == 0:
        return AggregateResult(experiment_name=experiment_name)

    was = [r.wa for r in fold_results]
    uas = [r.ua for r in fold_results]
    f1s = [r.macro_f1 for r in fold_results]

    # Per-class F1 aggregation
    all_labels = set()
    for r in fold_results:
        all_labels.update(r.per_class_f1.keys())

    pc_f1_mean = {}
    pc_f1_std = {}
    for label in sorted(all_labels):
        values = [r.per_class_f1.get(label, 0.0) for r in fold_results]
        pc_f1_mean[label] = float(np.mean(values))
        pc_f1_std[label] = float(np.std(values))

    # Average confusion matrix
    cms = []
    for r in fold_results:
        if r.confusion_matrix:
            cms.append(np.array(r.confusion_matrix, dtype=float))
    avg_cm = np.mean(cms, axis=0).tolist() if cms else []

    return AggregateResult(
        experiment_name=experiment_name,
        num_folds=n,
        wa_mean=float(np.mean(was)),
        wa_std=float(np.std(was)),
        ua_mean=float(np.mean(uas)),
        ua_std=float(np.std(uas)),
        macro_f1_mean=float(np.mean(f1s)),
        macro_f1_std=float(np.std(f1s)),
        per_class_f1_mean=pc_f1_mean,
        per_class_f1_std=pc_f1_std,
        avg_confusion_matrix=avg_cm,
        fold_results=[r.to_dict() for r in fold_results],
    )


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def save_results(result: AggregateResult | FoldResult, path: str | Path) -> None:
    """Save evaluation results to JSON.

    Args:
        result: An AggregateResult or FoldResult.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = result.to_dict() if hasattr(result, "to_dict") else asdict(result)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Results saved to {path}")


def load_results(path: str | Path) -> dict:
    """Load evaluation results from JSON.

    Args:
        path: Path to JSON results file.

    Returns:
        Dict of results.
    """
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def format_results_table(
    results: list[AggregateResult],
    format: str = "markdown",
) -> str:
    """Generate a formatted comparison table from multiple experiment results.

    Args:
        results: List of AggregateResult objects.
        format: ``"markdown"`` or ``"latex"``.

    Returns:
        Formatted table string.
    """
    if format == "latex":
        return _format_latex_table(results)
    return _format_markdown_table(results)


def _format_markdown_table(results: list[AggregateResult]) -> str:
    """Generate a Markdown table comparing experiment results."""
    lines = []
    header = "| Experiment | WA (%) | UA (%) | Macro-F1 (%) | Delta UA |"
    sep = "|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)

    # Use first result as baseline for delta computation
    baseline_ua = results[0].ua_mean if results else 0.0

    for r in results:
        delta = r.ua_mean - baseline_ua
        delta_str = f"{delta:+.1f}" if r != results[0] else "---"
        lines.append(
            f"| {r.experiment_name} "
            f"| {r.ua_mean * 100:.1f} +/- {r.ua_std * 100:.1f} "
            f"| {r.wa_mean * 100:.1f} +/- {r.wa_std * 100:.1f} "
            f"| {r.macro_f1_mean * 100:.1f} +/- {r.macro_f1_std * 100:.1f} "
            f"| {delta_str} |"
        )

    return "\n".join(lines)


def _format_latex_table(results: list[AggregateResult]) -> str:
    """Generate a LaTeX table comparing experiment results."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation study results on IEMOCAP (5-fold LOSO-CV)}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Experiment & WA (\%) & UA (\%) & Macro-F1 (\%) & $\Delta$ UA \\")
    lines.append(r"\midrule")

    baseline_ua = results[0].ua_mean if results else 0.0

    for i, r in enumerate(results):
        delta = r.ua_mean - baseline_ua
        delta_str = f"{delta * 100:+.1f}" if i > 0 else "---"

        name = r.experiment_name.replace("_", r"\_")
        line = (
            f"{name} & "
            f"{r.wa_mean * 100:.1f} $\\pm$ {r.wa_std * 100:.1f} & "
            f"{r.ua_mean * 100:.1f} $\\pm$ {r.ua_std * 100:.1f} & "
            f"{r.macro_f1_mean * 100:.1f} $\\pm$ {r.macro_f1_std * 100:.1f} & "
            f"{delta_str} \\\\"
        )
        lines.append(line)

        # Add midrule after first row (baseline)
        if i == 0:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def format_confusion_matrix(
    cm: list[list[float]] | np.ndarray,
    label_names: Optional[list[str]] = None,
    format: str = "markdown",
) -> str:
    """Format a confusion matrix for display.

    Args:
        cm: Confusion matrix (num_classes x num_classes).
        label_names: Class names for row/column headers.
        format: ``"markdown"`` or ``"text"``.

    Returns:
        Formatted string.
    """
    if isinstance(cm, np.ndarray):
        cm = cm.tolist()

    if label_names is None:
        label_names = EMOTION_LABELS

    n = len(cm)
    lines = []

    if format == "markdown":
        # Header
        header = "| |" + "|".join(f" **{l}** " for l in label_names[:n]) + "|"
        sep = "|---|" + "|".join("---" for _ in range(n)) + "|"
        lines.append(header)
        lines.append(sep)
        for i in range(n):
            row = f"| **{label_names[i]}** |"
            row += "|".join(f" {cm[i][j]:.1f} " for j in range(n))
            row += "|"
            lines.append(row)
    else:
        # Plain text
        max_len = max(len(l) for l in label_names[:n])
        header = " " * (max_len + 2) + "  ".join(f"{l:>8s}" for l in label_names[:n])
        lines.append(header)
        for i in range(n):
            row = f"{label_names[i]:>{max_len}s}  "
            row += "  ".join(f"{cm[i][j]:8.1f}" for j in range(n))
            lines.append(row)

    return "\n".join(lines)
