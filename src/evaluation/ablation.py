"""Ablation study runner for MERA.

Systematically tests each component's contribution by training and
evaluating model variants.  Supports all ablation experiments from
Section 9.2 of the architecture document:

Full system:
    MERA-Full

Pipeline ablations:
    Text-Only, Audio-Only

Sub-stream ablations (within Pipeline B):
    No-CNN-BiLSTM, No-ProsodicTCN, No-HuBERT, TCN-Only

Novel component ablations:
    No-AttentionEntropy, No-ModalityDropout, No-CrossModalAttention,
    No-GatedFusion

Baseline comparisons:
    OpenSMILE-SVM, OpenSMILE-MLP, Summary-Stats-TCN
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ablation experiment definitions
# ---------------------------------------------------------------------------

class AblationType(Enum):
    """All supported ablation experiment types."""

    MERA_FULL = "MERA-Full"
    TEXT_ONLY = "Text-Only"
    AUDIO_ONLY = "Audio-Only"
    NO_CNN_BILSTM = "No-CNN-BiLSTM"
    NO_PROSODIC_TCN = "No-ProsodicTCN"
    NO_HUBERT = "No-HuBERT"
    TCN_ONLY = "TCN-Only"
    NO_ATTENTION_ENTROPY = "No-AttentionEntropy"
    NO_MODALITY_DROPOUT = "No-ModalityDropout"
    NO_CROSS_MODAL_ATTENTION = "No-CrossModalAttention"
    NO_GATED_FUSION = "No-GatedFusion"
    OPENSMILE_SVM = "OpenSMILE-SVM"
    OPENSMILE_MLP = "OpenSMILE-MLP"
    SUMMARY_STATS_TCN = "Summary-Stats-TCN"


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment.

    Attributes:
        name: Human-readable experiment name.
        ablation_type: The type of ablation.
        description: What is being tested.
        model_modifications: Dict describing how to modify the model or config.
        is_baseline: Whether this is a non-neural baseline (SVM, MLP).
        requires_full_model: Whether the experiment uses the full MERA model.
    """

    name: str
    ablation_type: AblationType
    description: str
    model_modifications: dict[str, Any] = field(default_factory=dict)
    is_baseline: bool = False
    requires_full_model: bool = True


# ---------------------------------------------------------------------------
# Registry of all ablation experiments
# ---------------------------------------------------------------------------

ALL_ABLATIONS: dict[str, AblationConfig] = {
    "MERA-Full": AblationConfig(
        name="MERA-Full",
        ablation_type=AblationType.MERA_FULL,
        description="Complete system with all components",
        model_modifications={},
    ),
    "Text-Only": AblationConfig(
        name="Text-Only",
        ablation_type=AblationType.TEXT_ONLY,
        description="Pipeline A alone (no audio)",
        model_modifications={"mode": "text_only"},
        requires_full_model=False,
    ),
    "Audio-Only": AblationConfig(
        name="Audio-Only",
        ablation_type=AblationType.AUDIO_ONLY,
        description="Pipeline B alone (no text)",
        model_modifications={"mode": "audio_only"},
        requires_full_model=False,
    ),
    "No-CNN-BiLSTM": AblationConfig(
        name="No-CNN-BiLSTM",
        ablation_type=AblationType.NO_CNN_BILSTM,
        description="Remove Stream 1, keep Streams 2+3",
        model_modifications={"disable_streams": ["stream1"]},
    ),
    "No-ProsodicTCN": AblationConfig(
        name="No-ProsodicTCN",
        ablation_type=AblationType.NO_PROSODIC_TCN,
        description="Remove Stream 2 (novel TCN), keep Streams 1+3",
        model_modifications={"disable_streams": ["stream2"]},
    ),
    "No-HuBERT": AblationConfig(
        name="No-HuBERT",
        ablation_type=AblationType.NO_HUBERT,
        description="Remove Stream 3, keep Streams 1+2",
        model_modifications={"disable_streams": ["stream3"]},
    ),
    "TCN-Only": AblationConfig(
        name="TCN-Only",
        ablation_type=AblationType.TCN_ONLY,
        description="Only the prosodic TCN stream (tests novel contribution in isolation)",
        model_modifications={"disable_streams": ["stream1", "stream3"]},
    ),
    "No-AttentionEntropy": AblationConfig(
        name="No-AttentionEntropy",
        ablation_type=AblationType.NO_ATTENTION_ENTROPY,
        description="Remove attention entropy signal from fusion gate",
        model_modifications={"zero_entropy": True},
    ),
    "No-ModalityDropout": AblationConfig(
        name="No-ModalityDropout",
        ablation_type=AblationType.NO_MODALITY_DROPOUT,
        description="Train without modality dropout",
        model_modifications={"modality_dropout_prob": 0.0},
    ),
    "No-CrossModalAttention": AblationConfig(
        name="No-CrossModalAttention",
        ablation_type=AblationType.NO_CROSS_MODAL_ATTENTION,
        description="Simple concatenation instead of cross-modal attention",
        model_modifications={"bypass_cross_attention": True},
    ),
    "No-GatedFusion": AblationConfig(
        name="No-GatedFusion",
        ablation_type=AblationType.NO_GATED_FUSION,
        description="Equal-weight average instead of learned gating",
        model_modifications={"equal_weight_fusion": True},
    ),
    "OpenSMILE-SVM": AblationConfig(
        name="OpenSMILE-SVM",
        ablation_type=AblationType.OPENSMILE_SVM,
        description="Traditional eGeMAPS features -> SVM baseline",
        model_modifications={"classifier": "svm"},
        is_baseline=True,
        requires_full_model=False,
    ),
    "OpenSMILE-MLP": AblationConfig(
        name="OpenSMILE-MLP",
        ablation_type=AblationType.OPENSMILE_MLP,
        description="eGeMAPS features -> 3-layer MLP baseline",
        model_modifications={"classifier": "mlp"},
        is_baseline=True,
        requires_full_model=False,
    ),
    "Summary-Stats-TCN": AblationConfig(
        name="Summary-Stats-TCN",
        ablation_type=AblationType.SUMMARY_STATS_TCN,
        description="Traditional summary statistics instead of prosodic contours in TCN",
        model_modifications={"use_summary_stats": True},
    ),
}


def get_ablation_config(name: str) -> AblationConfig:
    """Get an ablation config by name.

    Args:
        name: Experiment name (e.g. "MERA-Full", "No-ProsodicTCN").

    Returns:
        The corresponding AblationConfig.

    Raises:
        KeyError: If the name is not registered.
    """
    if name not in ALL_ABLATIONS:
        available = ", ".join(sorted(ALL_ABLATIONS.keys()))
        raise KeyError(f"Unknown ablation '{name}'. Available: {available}")
    return ALL_ABLATIONS[name]


def list_ablation_names() -> list[str]:
    """Return all registered ablation experiment names."""
    return list(ALL_ABLATIONS.keys())


# ---------------------------------------------------------------------------
# Model modification functions
# ---------------------------------------------------------------------------

def apply_stream_ablation(
    model: nn.Module,
    disabled_streams: list[str],
) -> nn.Module:
    """Replace specified Pipeline B streams with zero-output wrappers.

    Args:
        model: A MERAModel or AudioEmotionHead instance.
        disabled_streams: List of stream names to disable
            (``"stream1"``, ``"stream2"``, ``"stream3"``).

    Returns:
        The modified model (in-place).
    """
    # Get the pipeline_b module (handle both MERAModel and standalone)
    if hasattr(model, "pipeline_b"):
        pipeline_b = model.pipeline_b
    elif hasattr(model, "stream1"):
        pipeline_b = model
    else:
        logger.warning("Cannot apply stream ablation: model has no pipeline_b")
        return model

    for stream_name in disabled_streams:
        if hasattr(pipeline_b, stream_name):
            original = getattr(pipeline_b, stream_name)
            wrapper = ZeroOutputWrapper(original)
            setattr(pipeline_b, stream_name, wrapper)
            logger.info(f"Disabled {stream_name} (replaced with zero-output wrapper)")
        else:
            logger.warning(f"Stream '{stream_name}' not found in pipeline_b")

    return model


class ZeroOutputWrapper(nn.Module):
    """Wraps a module and returns zeros matching the original output shape.

    This allows ablation studies to disable specific streams without
    changing the model architecture, so the SE fusion module still
    receives the correct number of inputs.
    """

    def __init__(self, original_module: nn.Module):
        super().__init__()
        self.original_module = original_module
        # Freeze the original to save memory
        for p in self.original_module.parameters():
            p.requires_grad = False

    def forward(self, *args, **kwargs):
        # Run the original to get the output shape, then zero it
        with torch.no_grad():
            output = self.original_module(*args, **kwargs)
        return torch.zeros_like(output)


def apply_fusion_ablation(
    model: nn.Module,
    modification: dict,
) -> nn.Module:
    """Apply fusion-level ablations to a MERAModel.

    Supported modifications:
    - ``zero_entropy``: Zero out attention entropy before gated fusion.
    - ``bypass_cross_attention``: Skip cross-modal attention (pass-through).
    - ``equal_weight_fusion``: Replace gated fusion with equal averaging.
    - ``modality_dropout_prob``: Override the modality dropout probability.

    Args:
        model: A MERAModel instance.
        modification: Dict of modifications to apply.

    Returns:
        The modified model (in-place).
    """
    if modification.get("zero_entropy"):
        if hasattr(model, "gated_fusion"):
            model.gated_fusion = ZeroEntropyGatedFusion(model.gated_fusion)
            logger.info("Attention entropy zeroed in gated fusion")

    if modification.get("bypass_cross_attention"):
        if hasattr(model, "cross_modal_attention"):
            model.cross_modal_attention = IdentityCrossAttention()
            logger.info("Cross-modal attention bypassed (identity pass-through)")

    if modification.get("equal_weight_fusion"):
        if hasattr(model, "gated_fusion"):
            model.gated_fusion = EqualWeightFusion()
            logger.info("Gated fusion replaced with equal-weight averaging")

    if "modality_dropout_prob" in modification:
        if hasattr(model, "modality_dropout_prob"):
            model.modality_dropout_prob = modification["modality_dropout_prob"]
            logger.info(f"Modality dropout set to {model.modality_dropout_prob}")

    return model


class ZeroEntropyGatedFusion(nn.Module):
    """Wraps GatedFusion but zeros out the entropy input."""

    def __init__(self, original_fusion: nn.Module):
        super().__init__()
        self.original_fusion = original_fusion

    def forward(self, text_repr, audio_repr, attention_entropy):
        zeroed_entropy = torch.zeros_like(attention_entropy)
        return self.original_fusion(text_repr, audio_repr, zeroed_entropy)


class IdentityCrossAttention(nn.Module):
    """Identity pass-through replacing cross-modal attention."""

    def forward(self, text_repr, audio_repr):
        return text_repr, audio_repr


class EqualWeightFusion(nn.Module):
    """Equal-weight average fusion replacing learned gating."""

    def forward(self, text_repr, audio_repr, attention_entropy):
        fused = (text_repr + audio_repr) / 2.0
        batch_size = text_repr.shape[0]
        gate_weights = torch.full(
            (batch_size, 2), 0.5,
            device=text_repr.device, dtype=text_repr.dtype,
        )
        return fused, gate_weights


# ---------------------------------------------------------------------------
# OpenSMILE baseline models
# ---------------------------------------------------------------------------

class OpenSMILEMLPBaseline(nn.Module):
    """3-layer MLP baseline on eGeMAPS features (88-d)."""

    def __init__(
        self,
        input_dim: int = 88,
        hidden_dim: int = 128,
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(batch, 88)`` eGeMAPS features.

        Returns:
            logits: ``(batch, num_classes)``
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# Ablation application dispatcher
# ---------------------------------------------------------------------------

def apply_ablation(
    model: nn.Module,
    ablation_config: AblationConfig,
) -> nn.Module:
    """Apply an ablation configuration to a model.

    This is the main entry point for modifying a model for a specific
    ablation experiment.

    Args:
        model: The model to modify.
        ablation_config: The ablation configuration.

    Returns:
        The modified model.
    """
    mods = ablation_config.model_modifications

    # Stream ablations
    if "disable_streams" in mods:
        model = apply_stream_ablation(model, mods["disable_streams"])

    # Fusion ablations
    fusion_keys = {"zero_entropy", "bypass_cross_attention",
                   "equal_weight_fusion", "modality_dropout_prob"}
    fusion_mods = {k: v for k, v in mods.items() if k in fusion_keys}
    if fusion_mods:
        model = apply_fusion_ablation(model, fusion_mods)

    return model


def build_ablation_comparison_table(
    results: dict[str, dict],
) -> str:
    """Build a comparison table from ablation experiment results.

    Args:
        results: Dict mapping experiment name to a dict with keys
            ``wa_mean``, ``wa_std``, ``ua_mean``, ``ua_std``,
            ``macro_f1_mean``, ``macro_f1_std``.

    Returns:
        Formatted Markdown table string.
    """
    lines = []
    header = "| Experiment | WA (%) | UA (%) | Macro-F1 (%) | Delta UA |"
    sep = "|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)

    baseline_ua = None
    for name, r in results.items():
        ua = r.get("ua_mean", 0.0)
        if baseline_ua is None:
            baseline_ua = ua

        delta = ua - baseline_ua
        delta_str = f"{delta * 100:+.1f}" if name != list(results.keys())[0] else "---"

        lines.append(
            f"| {name} "
            f"| {r.get('wa_mean', 0) * 100:.1f} +/- {r.get('wa_std', 0) * 100:.1f} "
            f"| {ua * 100:.1f} +/- {r.get('ua_std', 0) * 100:.1f} "
            f"| {r.get('macro_f1_mean', 0) * 100:.1f} +/- {r.get('macro_f1_std', 0) * 100:.1f} "
            f"| {delta_str} |"
        )

    return "\n".join(lines)
