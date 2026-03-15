#!/usr/bin/env python3
"""Tests for MERA evaluation, ablation, and cross-dataset infrastructure.

Covers:
- Evaluator computes metrics correctly on known predictions
- FoldResult and AggregateResult data structures
- Aggregate fold results (mean +/- std)
- Results save/load JSON round-trip
- Table formatting (Markdown and LaTeX)
- Confusion matrix formatting
- Ablation config registry (all 14 experiments registered)
- Ablation model modifications (stream disabling, fusion ablations)
- ZeroOutputWrapper, IdentityCrossAttention, EqualWeightFusion
- ZeroEntropyGatedFusion
- OpenSMILE MLP baseline model
- Cross-dataset experiment registry
- Cross-dataset split construction
- Cross-dataset table formatting
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import (
    Evaluator,
    FoldResult,
    AggregateResult,
    aggregate_fold_results,
    save_results,
    load_results,
    format_results_table,
    format_confusion_matrix,
)
from src.evaluation.ablation import (
    AblationType,
    AblationConfig,
    ALL_ABLATIONS,
    get_ablation_config,
    list_ablation_names,
    apply_stream_ablation,
    apply_fusion_ablation,
    apply_ablation,
    ZeroOutputWrapper,
    IdentityCrossAttention,
    EqualWeightFusion,
    ZeroEntropyGatedFusion,
    OpenSMILEMLPBaseline,
    build_ablation_comparison_table,
)
from src.evaluation.cross_dataset import (
    CrossDatasetExperiment,
    CrossDatasetResult,
    ALL_CROSS_DATASET_EXPERIMENTS,
    get_cross_dataset_experiment,
    list_cross_dataset_experiments,
    build_cross_dataset_split,
    format_cross_dataset_table,
)
from src.training.losses import PipelineLoss


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SimpleModel(nn.Module):
    """Tiny model for evaluator tests."""

    def __init__(self, input_dim=8, num_classes=4):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def simple_forward_fn(model, batch, device):
    """Forward function for the simple model."""
    x = batch[0].to(device)
    targets = batch[1].to(device)
    logits = model(x)
    loss_fn = simple_forward_fn._loss_fn
    loss_dict = loss_fn(logits, targets)
    return loss_dict, logits, targets


# ============================================================================
# Tests: Evaluator
# ============================================================================

class TestEvaluator:
    """Tests for the Evaluator class."""

    def _make_loader(self, n_samples=32, input_dim=8, num_classes=4, batch_size=8):
        X = torch.randn(n_samples, input_dim)
        y = torch.randint(0, num_classes, (n_samples,))
        ds = TensorDataset(X, y)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)

    def test_evaluate_returns_fold_result(self, device):
        """Evaluator.evaluate() should return a FoldResult."""
        model = SimpleModel().to(device)
        loss_fn = PipelineLoss(label_smoothing=0.0)
        simple_forward_fn._loss_fn = loss_fn

        evaluator = Evaluator(
            model=model,
            device=device,
            forward_fn=simple_forward_fn,
            num_classes=4,
        )
        loader = self._make_loader()
        result = evaluator.evaluate(loader)

        assert isinstance(result, FoldResult)
        assert 0.0 <= result.wa <= 1.0
        assert 0.0 <= result.ua <= 1.0
        assert 0.0 <= result.macro_f1 <= 1.0
        assert result.total_samples == 32

    def test_evaluate_perfect_model(self, device):
        """A model that returns perfect predictions should get WA=UA=1."""
        # Create a dataset where we know the mapping
        X = torch.eye(4).repeat(5, 1)  # 20 samples, each row repeated
        y = torch.tensor([0, 1, 2, 3] * 5)

        # Build a model that maps identity inputs to correct logits
        model = nn.Linear(4, 4, bias=False)
        # Set weights to identity * large_value for strong predictions
        with torch.no_grad():
            model.weight.copy_(torch.eye(4) * 10)
        model = model.to(device)

        loss_fn = PipelineLoss(label_smoothing=0.0)
        simple_forward_fn._loss_fn = loss_fn

        ds = TensorDataset(X, y)
        loader = DataLoader(ds, batch_size=4, shuffle=False)

        evaluator = Evaluator(
            model=model,
            device=device,
            forward_fn=simple_forward_fn,
            num_classes=4,
        )
        result = evaluator.evaluate(loader)

        assert result.wa == pytest.approx(1.0)
        assert result.ua == pytest.approx(1.0)
        assert result.macro_f1 == pytest.approx(1.0)

    def test_evaluate_confusion_matrix_shape(self, device):
        """Confusion matrix should be 4x4."""
        model = SimpleModel().to(device)
        loss_fn = PipelineLoss(label_smoothing=0.0)
        simple_forward_fn._loss_fn = loss_fn

        evaluator = Evaluator(
            model=model, device=device,
            forward_fn=simple_forward_fn, num_classes=4,
        )
        result = evaluator.evaluate(self._make_loader())
        cm = result.confusion_matrix
        assert len(cm) == 4
        assert all(len(row) == 4 for row in cm)


# ============================================================================
# Tests: FoldResult and AggregateResult
# ============================================================================

class TestResultContainers:
    """Tests for FoldResult and AggregateResult data containers."""

    def test_fold_result_to_dict(self):
        result = FoldResult(
            fold=1, wa=0.75, ua=0.72, macro_f1=0.70,
            per_class_f1={"angry": 0.8, "happy": 0.7},
            total_samples=100,
        )
        d = result.to_dict()
        assert d["fold"] == 1
        assert d["wa"] == 0.75
        assert d["total_samples"] == 100

    def test_aggregate_result_to_dict(self):
        result = AggregateResult(
            experiment_name="MERA-Full",
            num_folds=5,
            wa_mean=0.75,
            wa_std=0.03,
        )
        d = result.to_dict()
        assert d["experiment_name"] == "MERA-Full"
        assert d["num_folds"] == 5

    def test_aggregate_fold_results(self):
        """aggregate_fold_results should compute correct mean and std."""
        fold_results = [
            FoldResult(fold=1, wa=0.70, ua=0.68, macro_f1=0.65,
                       per_class_f1={"angry": 0.7, "happy": 0.6}),
            FoldResult(fold=2, wa=0.80, ua=0.78, macro_f1=0.75,
                       per_class_f1={"angry": 0.8, "happy": 0.7}),
            FoldResult(fold=3, wa=0.75, ua=0.73, macro_f1=0.70,
                       per_class_f1={"angry": 0.75, "happy": 0.65}),
        ]
        agg = aggregate_fold_results(fold_results, experiment_name="Test")

        assert agg.experiment_name == "Test"
        assert agg.num_folds == 3
        assert agg.wa_mean == pytest.approx(0.75, abs=1e-6)
        assert agg.ua_mean == pytest.approx(np.mean([0.68, 0.78, 0.73]), abs=1e-6)
        assert agg.wa_std == pytest.approx(np.std([0.70, 0.80, 0.75]), abs=1e-6)
        assert agg.ua_std == pytest.approx(np.std([0.68, 0.78, 0.73]), abs=1e-6)

    def test_aggregate_per_class_f1(self):
        """Per-class F1 should be aggregated correctly."""
        fold_results = [
            FoldResult(fold=1, per_class_f1={"angry": 0.8, "happy": 0.6}),
            FoldResult(fold=2, per_class_f1={"angry": 0.7, "happy": 0.7}),
        ]
        agg = aggregate_fold_results(fold_results)

        assert agg.per_class_f1_mean["angry"] == pytest.approx(0.75)
        assert agg.per_class_f1_mean["happy"] == pytest.approx(0.65)

    def test_aggregate_confusion_matrices(self):
        """Confusion matrices should be averaged across folds."""
        fold_results = [
            FoldResult(fold=1, confusion_matrix=[[10, 2], [3, 5]]),
            FoldResult(fold=2, confusion_matrix=[[8, 4], [1, 7]]),
        ]
        agg = aggregate_fold_results(fold_results)
        avg_cm = agg.avg_confusion_matrix

        assert avg_cm[0][0] == pytest.approx(9.0)
        assert avg_cm[0][1] == pytest.approx(3.0)
        assert avg_cm[1][0] == pytest.approx(2.0)
        assert avg_cm[1][1] == pytest.approx(6.0)

    def test_aggregate_empty_folds(self):
        """Aggregating empty list should return empty result."""
        agg = aggregate_fold_results([], experiment_name="Empty")
        assert agg.num_folds == 0
        assert agg.wa_mean == 0.0


# ============================================================================
# Tests: Results I/O
# ============================================================================

class TestResultsIO:
    """Tests for save/load results."""

    def test_save_and_load_roundtrip(self):
        """Saving and loading should preserve all data."""
        result = AggregateResult(
            experiment_name="Test",
            num_folds=5,
            wa_mean=0.75,
            wa_std=0.03,
            ua_mean=0.72,
            ua_std=0.04,
            macro_f1_mean=0.70,
            macro_f1_std=0.05,
            per_class_f1_mean={"angry": 0.8, "happy": 0.7},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.json"
            save_results(result, path)
            assert path.exists()

            loaded = load_results(path)
            assert loaded["experiment_name"] == "Test"
            assert loaded["wa_mean"] == pytest.approx(0.75)
            assert loaded["ua_mean"] == pytest.approx(0.72)

    def test_save_fold_result(self):
        """FoldResult should also be saveable."""
        result = FoldResult(fold=1, wa=0.75, ua=0.72)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fold1.json"
            save_results(result, path)
            loaded = load_results(path)
            assert loaded["fold"] == 1
            assert loaded["wa"] == 0.75

    def test_save_creates_directories(self):
        """save_results should create parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "deep" / "results.json"
            result = FoldResult(fold=0)
            save_results(result, path)
            assert path.exists()


# ============================================================================
# Tests: Table Formatting
# ============================================================================

class TestTableFormatting:
    """Tests for table formatting functions."""

    def _make_agg_results(self):
        return [
            AggregateResult(
                experiment_name="MERA-Full", num_folds=5,
                wa_mean=0.75, wa_std=0.03,
                ua_mean=0.72, ua_std=0.04,
                macro_f1_mean=0.70, macro_f1_std=0.05,
            ),
            AggregateResult(
                experiment_name="Text-Only", num_folds=5,
                wa_mean=0.65, wa_std=0.04,
                ua_mean=0.62, ua_std=0.05,
                macro_f1_mean=0.60, macro_f1_std=0.06,
            ),
        ]

    def test_markdown_table_format(self):
        """Markdown table should have proper structure."""
        results = self._make_agg_results()
        table = format_results_table(results, format="markdown")

        assert "|" in table
        assert "MERA-Full" in table
        assert "Text-Only" in table
        assert "---" in table  # separator row

        lines = table.strip().split("\n")
        assert len(lines) == 4  # header + sep + 2 data rows

    def test_latex_table_format(self):
        """LaTeX table should have proper structure."""
        results = self._make_agg_results()
        table = format_results_table(results, format="latex")

        assert r"\begin{table}" in table
        assert r"\end{table}" in table
        assert r"\toprule" in table
        assert r"\bottomrule" in table
        assert "MERA-Full" in table

    def test_delta_ua_computation(self):
        """Delta UA should be computed relative to first experiment."""
        results = self._make_agg_results()
        table = format_results_table(results, format="markdown")

        # First experiment should have "---" as delta
        assert "---" in table
        # Second experiment should have negative delta
        assert "-" in table.split("\n")[-1]  # Text-Only is worse

    def test_confusion_matrix_markdown(self):
        """Confusion matrix formatting should produce valid table."""
        cm = [[50, 5, 3, 2], [4, 40, 6, 0], [2, 3, 45, 5], [1, 2, 4, 38]]
        table = format_confusion_matrix(cm, format="markdown")

        assert "|" in table
        assert "angry" in table
        assert "happy" in table
        assert "sad" in table
        assert "neutral" in table

    def test_confusion_matrix_text(self):
        """Plain text confusion matrix format."""
        cm = np.array([[10, 2], [3, 5]])
        table = format_confusion_matrix(cm, label_names=["a", "b"], format="text")
        assert "a" in table
        assert "b" in table
        assert "10.0" in table

    def test_confusion_matrix_numpy_input(self):
        """Should accept numpy array input."""
        cm = np.array([[10, 2], [3, 5]])
        table = format_confusion_matrix(cm, label_names=["a", "b"])
        assert "10.0" in table


# ============================================================================
# Tests: Ablation Config Registry
# ============================================================================

class TestAblationRegistry:
    """Tests for the ablation experiment registry."""

    def test_all_14_experiments_registered(self):
        """All 14 ablation experiments from the architecture doc should be registered."""
        names = list_ablation_names()
        assert len(names) == 14

        expected = {
            "MERA-Full", "Text-Only", "Audio-Only",
            "No-CNN-BiLSTM", "No-ProsodicTCN", "No-HuBERT", "TCN-Only",
            "No-AttentionEntropy", "No-ModalityDropout",
            "No-CrossModalAttention", "No-GatedFusion",
            "OpenSMILE-SVM", "OpenSMILE-MLP", "Summary-Stats-TCN",
        }
        assert set(names) == expected

    def test_get_ablation_config(self):
        """get_ablation_config should return the correct config."""
        cfg = get_ablation_config("MERA-Full")
        assert cfg.name == "MERA-Full"
        assert cfg.ablation_type == AblationType.MERA_FULL
        assert not cfg.is_baseline

    def test_unknown_ablation_raises(self):
        """Requesting an unknown ablation should raise KeyError."""
        with pytest.raises(KeyError, match="Unknown ablation"):
            get_ablation_config("NonExistent")

    def test_baseline_experiments_flagged(self):
        """OpenSMILE baselines should have is_baseline=True."""
        svm_cfg = get_ablation_config("OpenSMILE-SVM")
        mlp_cfg = get_ablation_config("OpenSMILE-MLP")
        assert svm_cfg.is_baseline is True
        assert mlp_cfg.is_baseline is True
        assert svm_cfg.requires_full_model is False

    def test_pipeline_ablation_configs(self):
        """Text-Only and Audio-Only should have correct mode."""
        text_cfg = get_ablation_config("Text-Only")
        assert text_cfg.model_modifications["mode"] == "text_only"

        audio_cfg = get_ablation_config("Audio-Only")
        assert audio_cfg.model_modifications["mode"] == "audio_only"

    def test_stream_ablation_configs(self):
        """Sub-stream ablations should list correct streams to disable."""
        no_cnn = get_ablation_config("No-CNN-BiLSTM")
        assert "stream1" in no_cnn.model_modifications["disable_streams"]

        no_tcn = get_ablation_config("No-ProsodicTCN")
        assert "stream2" in no_tcn.model_modifications["disable_streams"]

        no_hub = get_ablation_config("No-HuBERT")
        assert "stream3" in no_hub.model_modifications["disable_streams"]

        tcn_only = get_ablation_config("TCN-Only")
        assert set(tcn_only.model_modifications["disable_streams"]) == {"stream1", "stream3"}

    def test_fusion_ablation_configs(self):
        """Fusion ablation configs should have the right modification flags."""
        no_entropy = get_ablation_config("No-AttentionEntropy")
        assert no_entropy.model_modifications.get("zero_entropy") is True

        no_dropout = get_ablation_config("No-ModalityDropout")
        assert no_dropout.model_modifications.get("modality_dropout_prob") == 0.0

        no_xattn = get_ablation_config("No-CrossModalAttention")
        assert no_xattn.model_modifications.get("bypass_cross_attention") is True

        no_gate = get_ablation_config("No-GatedFusion")
        assert no_gate.model_modifications.get("equal_weight_fusion") is True

    def test_each_ablation_has_description(self):
        """Every ablation should have a non-empty description."""
        for name in list_ablation_names():
            cfg = get_ablation_config(name)
            assert len(cfg.description) > 0, f"{name} has empty description"

    def test_each_ablation_has_type(self):
        """Every ablation should have a valid AblationType."""
        for name in list_ablation_names():
            cfg = get_ablation_config(name)
            assert isinstance(cfg.ablation_type, AblationType)


# ============================================================================
# Tests: Ablation Model Modifications
# ============================================================================

class DummyStream(nn.Module):
    """Dummy stream module for testing."""

    def __init__(self, output_dim=16):
        super().__init__()
        self.fc = nn.Linear(8, output_dim)

    def forward(self, x, mask=None):
        return self.fc(x)


class DummyPipelineB(nn.Module):
    """Dummy Pipeline B with three streams."""

    def __init__(self):
        super().__init__()
        self.stream1 = DummyStream(16)
        self.stream2 = DummyStream(8)
        self.stream3 = DummyStream(16)

    def forward(self, x):
        s1 = self.stream1(x)
        s2 = self.stream2(x)
        s3 = self.stream3(x)
        return torch.cat([s1, s2, s3], dim=-1)


class DummyGatedFusion(nn.Module):
    """Dummy gated fusion for testing."""

    def forward(self, text_repr, audio_repr, attention_entropy):
        fused = text_repr + audio_repr
        gate_weights = torch.ones(text_repr.shape[0], 2, device=text_repr.device) * 0.5
        return fused, gate_weights


class TestStreamAblation:
    """Tests for stream-level ablations."""

    def test_zero_output_wrapper(self):
        """ZeroOutputWrapper should return zeros."""
        stream = DummyStream(16)
        wrapper = ZeroOutputWrapper(stream)
        x = torch.randn(4, 8)
        output = wrapper(x)
        assert output.shape == (4, 16)
        assert torch.all(output == 0)

    def test_zero_output_wrapper_freezes_params(self):
        """Wrapped module parameters should be frozen."""
        stream = DummyStream(16)
        wrapper = ZeroOutputWrapper(stream)
        for p in wrapper.original_module.parameters():
            assert not p.requires_grad

    def test_apply_stream_ablation(self):
        """apply_stream_ablation should replace streams with zero wrappers."""
        pipeline_b = DummyPipelineB()

        # Store original output
        x = torch.randn(4, 8)
        original_out = pipeline_b(x)

        # Disable stream1
        apply_stream_ablation(pipeline_b, ["stream1"])
        assert isinstance(pipeline_b.stream1, ZeroOutputWrapper)

        # stream2 and stream3 should still be original
        assert isinstance(pipeline_b.stream2, DummyStream)
        assert isinstance(pipeline_b.stream3, DummyStream)

    def test_apply_multiple_streams(self):
        """Should be able to disable multiple streams."""
        pipeline_b = DummyPipelineB()
        apply_stream_ablation(pipeline_b, ["stream1", "stream3"])

        assert isinstance(pipeline_b.stream1, ZeroOutputWrapper)
        assert isinstance(pipeline_b.stream2, DummyStream)
        assert isinstance(pipeline_b.stream3, ZeroOutputWrapper)

    def test_ablation_on_model_with_pipeline_b(self):
        """Should work on a model that has a pipeline_b attribute."""
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.pipeline_b = DummyPipelineB()

        model = DummyModel()
        apply_stream_ablation(model, ["stream2"])
        assert isinstance(model.pipeline_b.stream2, ZeroOutputWrapper)


class TestFusionAblation:
    """Tests for fusion-level ablations."""

    def test_identity_cross_attention(self):
        """IdentityCrossAttention should pass through inputs unchanged."""
        ica = IdentityCrossAttention()
        text = torch.randn(4, 256)
        audio = torch.randn(4, 256)
        t_out, a_out = ica(text, audio)

        torch.testing.assert_close(t_out, text)
        torch.testing.assert_close(a_out, audio)

    def test_equal_weight_fusion(self):
        """EqualWeightFusion should average text and audio equally."""
        ewf = EqualWeightFusion()
        text = torch.ones(4, 256) * 2.0
        audio = torch.ones(4, 256) * 4.0
        entropy = torch.randn(4, 12)

        fused, gate_weights = ewf(text, audio, entropy)

        expected_fused = torch.ones(4, 256) * 3.0  # (2+4)/2
        torch.testing.assert_close(fused, expected_fused)
        assert gate_weights.shape == (4, 2)
        assert torch.all(gate_weights == 0.5)

    def test_zero_entropy_gated_fusion(self):
        """ZeroEntropyGatedFusion should zero out entropy before passing."""
        original = DummyGatedFusion()
        wrapper = ZeroEntropyGatedFusion(original)

        text = torch.randn(4, 256)
        audio = torch.randn(4, 256)
        entropy = torch.randn(4, 12)

        fused, gate_weights = wrapper(text, audio, entropy)
        assert fused.shape == (4, 256)
        assert gate_weights.shape == (4, 2)

    def test_apply_fusion_ablation_bypass_cross_attention(self):
        """Applying bypass_cross_attention should replace with identity."""
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.cross_modal_attention = nn.Linear(10, 10)
                self.gated_fusion = DummyGatedFusion()

        model = DummyModel()
        apply_fusion_ablation(model, {"bypass_cross_attention": True})
        assert isinstance(model.cross_modal_attention, IdentityCrossAttention)

    def test_apply_fusion_ablation_equal_weight(self):
        """Applying equal_weight_fusion should replace gated fusion."""
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.gated_fusion = DummyGatedFusion()

        model = DummyModel()
        apply_fusion_ablation(model, {"equal_weight_fusion": True})
        assert isinstance(model.gated_fusion, EqualWeightFusion)

    def test_apply_fusion_ablation_modality_dropout(self):
        """Should set modality_dropout_prob on the model."""
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.modality_dropout_prob = 0.3

        model = DummyModel()
        apply_fusion_ablation(model, {"modality_dropout_prob": 0.0})
        assert model.modality_dropout_prob == 0.0


class TestApplyAblation:
    """Tests for the apply_ablation dispatcher."""

    def test_mera_full_no_modification(self):
        """MERA-Full should not modify the model."""
        model = DummyPipelineB()
        cfg = get_ablation_config("MERA-Full")
        result = apply_ablation(model, cfg)
        # All streams should remain original
        assert isinstance(result.stream1, DummyStream)
        assert isinstance(result.stream2, DummyStream)
        assert isinstance(result.stream3, DummyStream)

    def test_no_prosodic_tcn_ablation(self):
        """No-ProsodicTCN should disable stream2."""
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.pipeline_b = DummyPipelineB()

        model = DummyModel()
        cfg = get_ablation_config("No-ProsodicTCN")
        apply_ablation(model, cfg)
        assert isinstance(model.pipeline_b.stream2, ZeroOutputWrapper)


# ============================================================================
# Tests: OpenSMILE Baseline
# ============================================================================

class TestOpenSMILEBaseline:
    """Tests for the OpenSMILE MLP baseline model."""

    def test_mlp_forward_shape(self, device):
        """MLP baseline should produce correct output shape."""
        model = OpenSMILEMLPBaseline(input_dim=88, num_classes=4).to(device)
        x = torch.randn(8, 88, device=device)
        logits = model(x)
        assert logits.shape == (8, 4)

    def test_mlp_gradient_flow(self, device):
        """MLP should allow gradient flow."""
        model = OpenSMILEMLPBaseline().to(device)
        x = torch.randn(8, 88, device=device, requires_grad=True)
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        assert x.grad is not None

    def test_mlp_custom_dims(self, device):
        """MLP should work with custom dimensions."""
        model = OpenSMILEMLPBaseline(
            input_dim=64, hidden_dim=32, num_classes=3,
        ).to(device)
        x = torch.randn(4, 64, device=device)
        logits = model(x)
        assert logits.shape == (4, 3)


# ============================================================================
# Tests: Ablation Comparison Table
# ============================================================================

class TestAblationComparisonTable:
    """Tests for build_ablation_comparison_table."""

    def test_table_structure(self):
        """Table should have header, separator, and data rows."""
        results = {
            "MERA-Full": {"wa_mean": 0.75, "wa_std": 0.03, "ua_mean": 0.72,
                          "ua_std": 0.04, "macro_f1_mean": 0.70, "macro_f1_std": 0.05},
            "Text-Only": {"wa_mean": 0.65, "wa_std": 0.04, "ua_mean": 0.62,
                          "ua_std": 0.05, "macro_f1_mean": 0.60, "macro_f1_std": 0.06},
        }
        table = build_ablation_comparison_table(results)
        lines = table.strip().split("\n")
        assert len(lines) == 4  # header + sep + 2 data rows

    def test_delta_ua_in_table(self):
        """Delta UA column should show correct differences."""
        results = {
            "MERA-Full": {"wa_mean": 0.75, "wa_std": 0, "ua_mean": 0.72,
                          "ua_std": 0, "macro_f1_mean": 0.70, "macro_f1_std": 0},
            "Worse": {"wa_mean": 0.65, "wa_std": 0, "ua_mean": 0.62,
                      "ua_std": 0, "macro_f1_mean": 0.60, "macro_f1_std": 0},
        }
        table = build_ablation_comparison_table(results)
        assert "---" in table  # Baseline row
        assert "-10.0" in table  # Delta for Worse experiment


# ============================================================================
# Tests: Cross-Dataset Registry
# ============================================================================

class TestCrossDatasetRegistry:
    """Tests for cross-dataset experiment registry."""

    def test_all_experiments_registered(self):
        """All 3 cross-dataset experiments should be registered."""
        names = list_cross_dataset_experiments()
        assert len(names) == 3

        expected = {
            "Train-IEMOCAP-Test-MSP",
            "Train-IEMOCAP-Test-MOSEI",
            "Train-All-Test-MOSEI",
        }
        assert set(names) == expected

    def test_get_experiment(self):
        """Should return correct experiment config."""
        exp = get_cross_dataset_experiment("Train-IEMOCAP-Test-MSP")
        assert exp.train_datasets == ["iemocap"]
        assert exp.test_datasets == ["msp_improv"]

    def test_train_all_config(self):
        """Train-All-Test-MOSEI should train on 3 datasets."""
        exp = get_cross_dataset_experiment("Train-All-Test-MOSEI")
        assert exp.train_datasets == ["iemocap", "ravdess", "cremad"]
        assert exp.test_datasets == ["cmu_mosei"]

    def test_unknown_experiment_raises(self):
        """Requesting an unknown experiment should raise KeyError."""
        with pytest.raises(KeyError, match="Unknown cross-dataset"):
            get_cross_dataset_experiment("NonExistent")


# ============================================================================
# Tests: Cross-Dataset Split Construction
# ============================================================================

class TestCrossDatasetSplits:
    """Tests for build_cross_dataset_split."""

    def test_basic_split(self):
        """Should correctly split data by experiment config."""
        exp = CrossDatasetExperiment(
            name="Test",
            train_datasets=["ds_a"],
            test_datasets=["ds_b"],
        )
        available = {
            "ds_a": ["record_a1", "record_a2"],
            "ds_b": ["record_b1"],
        }
        split = build_cross_dataset_split(exp, available)

        assert len(split["train"]) == 2
        assert len(split["test"]) == 1
        assert split["train"] == ["record_a1", "record_a2"]
        assert split["test"] == ["record_b1"]

    def test_multi_dataset_train(self):
        """Training on multiple datasets should combine records."""
        exp = CrossDatasetExperiment(
            name="Test",
            train_datasets=["ds_a", "ds_b"],
            test_datasets=["ds_c"],
        )
        available = {
            "ds_a": ["a1", "a2"],
            "ds_b": ["b1", "b2", "b3"],
            "ds_c": ["c1"],
        }
        split = build_cross_dataset_split(exp, available)

        assert len(split["train"]) == 5  # 2 + 3
        assert len(split["test"]) == 1

    def test_missing_train_dataset_raises(self):
        """Should raise ValueError if a train dataset is missing."""
        exp = CrossDatasetExperiment(
            name="Test",
            train_datasets=["missing_ds"],
            test_datasets=["ds_b"],
        )
        available = {"ds_b": ["b1"]}

        with pytest.raises(ValueError, match="Training dataset.*not available"):
            build_cross_dataset_split(exp, available)

    def test_missing_test_dataset_raises(self):
        """Should raise ValueError if a test dataset is missing."""
        exp = CrossDatasetExperiment(
            name="Test",
            train_datasets=["ds_a"],
            test_datasets=["missing_ds"],
        )
        available = {"ds_a": ["a1"]}

        with pytest.raises(ValueError, match="Test dataset.*not available"):
            build_cross_dataset_split(exp, available)

    def test_non_list_data(self):
        """Should handle non-list data (e.g., DataFrames) gracefully."""
        exp = CrossDatasetExperiment(
            name="Test",
            train_datasets=["ds_a"],
            test_datasets=["ds_b"],
        )
        available = {
            "ds_a": "dataframe_placeholder",
            "ds_b": "another_placeholder",
        }
        split = build_cross_dataset_split(exp, available)
        assert len(split["train"]) == 1
        assert len(split["test"]) == 1


# ============================================================================
# Tests: Cross-Dataset Result
# ============================================================================

class TestCrossDatasetResult:
    """Tests for CrossDatasetResult."""

    def test_compute_gap(self):
        """Generalization gap should be within - cross."""
        result = CrossDatasetResult(
            experiment_name="Test",
            train_datasets=["iemocap"],
            test_datasets=["msp_improv"],
            ua=0.55,
            within_dataset_ua=0.72,
        )
        gap = result.compute_gap()
        assert gap == pytest.approx(0.17)
        assert result.generalization_gap == pytest.approx(0.17)

    def test_to_dict(self):
        """Should serialize to dict correctly."""
        result = CrossDatasetResult(
            experiment_name="Test",
            train_datasets=["iemocap"],
            test_datasets=["msp_improv"],
            wa=0.60, ua=0.55,
        )
        d = result.to_dict()
        assert d["experiment_name"] == "Test"
        assert d["wa"] == 0.60


# ============================================================================
# Tests: Cross-Dataset Table Formatting
# ============================================================================

class TestCrossDatasetTable:
    """Tests for cross-dataset table formatting."""

    def _make_results(self):
        r1 = CrossDatasetResult(
            experiment_name="Train-IEMOCAP-Test-MSP",
            train_datasets=["iemocap"],
            test_datasets=["msp_improv"],
            wa=0.55, ua=0.50,
            within_dataset_ua=0.72,
        )
        r1.compute_gap()

        r2 = CrossDatasetResult(
            experiment_name="Train-All-Test-MOSEI",
            train_datasets=["iemocap", "ravdess", "cremad"],
            test_datasets=["cmu_mosei"],
            wa=0.52, ua=0.48,
            within_dataset_ua=0.72,
        )
        r2.compute_gap()
        return [r1, r2]

    def test_markdown_format(self):
        """Markdown table should include all experiments."""
        results = self._make_results()
        table = format_cross_dataset_table(results, format="markdown")

        assert "|" in table
        assert "Train-IEMOCAP-Test-MSP" in table
        assert "Train-All-Test-MOSEI" in table
        assert "iemocap" in table

    def test_latex_format(self):
        """LaTeX table should have proper structure."""
        results = self._make_results()
        table = format_cross_dataset_table(results, format="latex")

        assert r"\begin{table}" in table
        assert r"\end{table}" in table
        assert "Train-IEMOCAP-Test-MSP" in table

    def test_gap_column(self):
        """Gap column should show correct values."""
        results = self._make_results()
        table = format_cross_dataset_table(results, format="markdown")
        # Gap = 0.72 - 0.50 = 0.22, displayed as +22.0
        assert "+22.0" in table
