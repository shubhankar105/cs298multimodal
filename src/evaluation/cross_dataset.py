"""Cross-dataset generalization testing for MERA.

Tests how well a model trained on one dataset configuration generalises
to unseen datasets.  Reports WA, UA, and the generalization gap compared
to within-dataset performance.

Experiments from Section 9.3 of the architecture document:
- Train-IEMOCAP-Test-MSP
- Train-IEMOCAP-Test-MOSEI
- Train-All-Test-MOSEI
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cross-dataset experiment definitions
# ---------------------------------------------------------------------------

@dataclass
class CrossDatasetExperiment:
    """Configuration for a cross-dataset generalization experiment.

    Attributes:
        name: Human-readable experiment name.
        train_datasets: List of dataset names used for training.
        test_datasets: List of dataset names used for testing.
        description: What this experiment tests.
    """

    name: str
    train_datasets: list[str]
    test_datasets: list[str]
    description: str = ""


# Registry of all cross-dataset experiments
ALL_CROSS_DATASET_EXPERIMENTS: dict[str, CrossDatasetExperiment] = {
    "Train-IEMOCAP-Test-MSP": CrossDatasetExperiment(
        name="Train-IEMOCAP-Test-MSP",
        train_datasets=["iemocap"],
        test_datasets=["msp_improv"],
        description="Train on IEMOCAP, test on MSP-IMPROV",
    ),
    "Train-IEMOCAP-Test-MOSEI": CrossDatasetExperiment(
        name="Train-IEMOCAP-Test-MOSEI",
        train_datasets=["iemocap"],
        test_datasets=["cmu_mosei"],
        description="Train on IEMOCAP, test on CMU-MOSEI",
    ),
    "Train-All-Test-MOSEI": CrossDatasetExperiment(
        name="Train-All-Test-MOSEI",
        train_datasets=["iemocap", "ravdess", "cremad"],
        test_datasets=["cmu_mosei"],
        description="Train on all available datasets, test on CMU-MOSEI",
    ),
}


def get_cross_dataset_experiment(name: str) -> CrossDatasetExperiment:
    """Get a cross-dataset experiment config by name.

    Args:
        name: Experiment name.

    Returns:
        CrossDatasetExperiment config.

    Raises:
        KeyError: If the name is not registered.
    """
    if name not in ALL_CROSS_DATASET_EXPERIMENTS:
        available = ", ".join(sorted(ALL_CROSS_DATASET_EXPERIMENTS.keys()))
        raise KeyError(
            f"Unknown cross-dataset experiment '{name}'. Available: {available}"
        )
    return ALL_CROSS_DATASET_EXPERIMENTS[name]


def list_cross_dataset_experiments() -> list[str]:
    """Return all registered cross-dataset experiment names."""
    return list(ALL_CROSS_DATASET_EXPERIMENTS.keys())


# ---------------------------------------------------------------------------
# Cross-dataset result containers
# ---------------------------------------------------------------------------

@dataclass
class CrossDatasetResult:
    """Results for a single cross-dataset experiment.

    Attributes:
        experiment_name: Name of the experiment.
        train_datasets: Datasets used for training.
        test_datasets: Datasets used for testing.
        wa: Weighted accuracy on the test set.
        ua: Unweighted accuracy on the test set.
        macro_f1: Macro-F1 on the test set.
        per_class_f1: Per-class F1 scores.
        within_dataset_ua: UA from within-dataset evaluation (for gap).
        generalization_gap: Difference between within-dataset and cross-dataset UA.
        total_samples: Number of test samples.
    """

    experiment_name: str
    train_datasets: list[str]
    test_datasets: list[str]
    wa: float = 0.0
    ua: float = 0.0
    macro_f1: float = 0.0
    per_class_f1: dict[str, float] = field(default_factory=dict)
    within_dataset_ua: float = 0.0
    generalization_gap: float = 0.0
    total_samples: int = 0

    def compute_gap(self) -> float:
        """Compute the generalization gap (within - cross)."""
        self.generalization_gap = self.within_dataset_ua - self.ua
        return self.generalization_gap

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Split construction
# ---------------------------------------------------------------------------

def build_cross_dataset_split(
    experiment: CrossDatasetExperiment,
    available_datasets: dict[str, Any],
) -> dict[str, list]:
    """Build train/test splits for a cross-dataset experiment.

    Args:
        experiment: The experiment configuration.
        available_datasets: Dict mapping dataset name to its records/dataframe.

    Returns:
        Dict with ``"train"`` and ``"test"`` keys, each containing a list
        of records from the respective datasets.

    Raises:
        ValueError: If a required dataset is not available.
    """
    train_records = []
    test_records = []

    for ds_name in experiment.train_datasets:
        if ds_name not in available_datasets:
            raise ValueError(
                f"Training dataset '{ds_name}' not available. "
                f"Available: {list(available_datasets.keys())}"
            )
        data = available_datasets[ds_name]
        if isinstance(data, list):
            train_records.extend(data)
        else:
            train_records.append(data)

    for ds_name in experiment.test_datasets:
        if ds_name not in available_datasets:
            raise ValueError(
                f"Test dataset '{ds_name}' not available. "
                f"Available: {list(available_datasets.keys())}"
            )
        data = available_datasets[ds_name]
        if isinstance(data, list):
            test_records.extend(data)
        else:
            test_records.append(data)

    return {"train": train_records, "test": test_records}


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def format_cross_dataset_table(
    results: list[CrossDatasetResult],
    format: str = "markdown",
) -> str:
    """Generate a formatted comparison table from cross-dataset results.

    Args:
        results: List of CrossDatasetResult objects.
        format: ``"markdown"`` or ``"latex"``.

    Returns:
        Formatted table string.
    """
    if format == "latex":
        return _format_cross_dataset_latex(results)
    return _format_cross_dataset_markdown(results)


def _format_cross_dataset_markdown(results: list[CrossDatasetResult]) -> str:
    """Format cross-dataset results as a Markdown table."""
    lines = []
    header = (
        "| Experiment | Train | Test | WA (%) | UA (%) | "
        "Within-DS UA (%) | Gap (pp) |"
    )
    sep = "|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)

    for r in results:
        train_str = "+".join(r.train_datasets)
        test_str = "+".join(r.test_datasets)
        gap = r.generalization_gap * 100
        lines.append(
            f"| {r.experiment_name} "
            f"| {train_str} "
            f"| {test_str} "
            f"| {r.wa * 100:.1f} "
            f"| {r.ua * 100:.1f} "
            f"| {r.within_dataset_ua * 100:.1f} "
            f"| {gap:+.1f} |"
        )

    return "\n".join(lines)


def _format_cross_dataset_latex(results: list[CrossDatasetResult]) -> str:
    """Format cross-dataset results as a LaTeX table."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Cross-dataset generalization results}")
    lines.append(r"\label{tab:cross_dataset}")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Experiment & Train $\rightarrow$ Test & WA (\%) & UA (\%) "
        r"& Within-DS UA (\%) & Gap (pp) \\"
    )
    lines.append(r"\midrule")

    for r in results:
        train_str = "+".join(r.train_datasets)
        test_str = "+".join(r.test_datasets)
        name = r.experiment_name.replace("_", r"\_").replace("-", r"-")
        gap = r.generalization_gap * 100
        lines.append(
            f"{name} & {train_str} $\\rightarrow$ {test_str} & "
            f"{r.wa * 100:.1f} & {r.ua * 100:.1f} & "
            f"{r.within_dataset_ua * 100:.1f} & {gap:+.1f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)
