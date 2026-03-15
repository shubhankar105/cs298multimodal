#!/usr/bin/env python3
"""Tests for MERA training infrastructure.

Covers:
- MERALoss with known inputs (all components + weighted total)
- PipelineLoss (simple cross-entropy)
- compute_class_weights (inverse_freq and effective_num)
- MetricTracker with deterministic predictions (WA, UA, F1, confusion matrix)
- Scheduler warmup ramp and cosine decay verification
- EarlyStopping patience logic and state management
- Trainer running one batch with a simple model
"""

from __future__ import annotations

import math
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

from src.training.losses import MERALoss, PipelineLoss, compute_class_weights
from src.training.metrics import (
    MetricTracker,
    compute_weighted_accuracy,
    compute_unweighted_accuracy,
    compute_f1_scores,
    compute_confusion_matrix,
)
from src.training.schedulers import build_cosine_warmup_scheduler
from src.training.early_stopping import EarlyStopping
from src.training.trainer import Trainer


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================================
# Tests: MERALoss
# ============================================================================

class TestMERALoss:
    """Tests for the multi-task MERALoss."""

    def test_output_keys(self):
        """MERALoss returns all expected loss component keys."""
        loss_fn = MERALoss(label_smoothing=0.0)
        outputs = {
            "final_logits": torch.randn(4, 4),
            "text_logits": torch.randn(4, 4),
            "audio_logits": torch.randn(4, 4),
        }
        targets = torch.tensor([0, 1, 2, 3])
        result = loss_fn(outputs, targets)

        assert "total" in result
        assert "primary" in result
        assert "aux_text" in result
        assert "aux_audio" in result
        assert "consistency" in result

    def test_all_losses_are_scalar_tensors(self):
        """Each loss component should be a scalar tensor."""
        loss_fn = MERALoss(label_smoothing=0.0)
        outputs = {
            "final_logits": torch.randn(8, 4),
            "text_logits": torch.randn(8, 4),
            "audio_logits": torch.randn(8, 4),
        }
        targets = torch.randint(0, 4, (8,))
        result = loss_fn(outputs, targets)

        for key, val in result.items():
            assert isinstance(val, torch.Tensor), f"{key} is not a tensor"
            assert val.dim() == 0, f"{key} is not scalar (shape={val.shape})"

    def test_weighted_total(self):
        """Total = lambda_primary*primary + lambda_aux_text*aux_text + lambda_aux_audio*aux_audio + lambda_consistency*consistency."""
        loss_fn = MERALoss(
            lambda_primary=1.0,
            lambda_aux_text=0.3,
            lambda_aux_audio=0.3,
            lambda_consistency=0.2,
            label_smoothing=0.0,
        )
        outputs = {
            "final_logits": torch.randn(8, 4),
            "text_logits": torch.randn(8, 4),
            "audio_logits": torch.randn(8, 4),
        }
        targets = torch.randint(0, 4, (8,))
        result = loss_fn(outputs, targets)

        expected = (
            1.0 * result["primary"]
            + 0.3 * result["aux_text"]
            + 0.3 * result["aux_audio"]
            + 0.2 * result["consistency"]
        )
        torch.testing.assert_close(result["total"], expected, rtol=1e-5, atol=1e-6)

    def test_perfect_prediction_low_loss(self):
        """When logits strongly predict the correct class, primary loss should be low."""
        loss_fn = MERALoss(label_smoothing=0.0)
        targets = torch.tensor([0, 1, 2, 3])
        # Strong predictions: 10 for correct class, -10 for others
        logits = torch.full((4, 4), -10.0)
        for i in range(4):
            logits[i, i] = 10.0

        outputs = {
            "final_logits": logits.clone(),
            "text_logits": logits.clone(),
            "audio_logits": logits.clone(),
        }
        result = loss_fn(outputs, targets)
        assert result["primary"].item() < 0.01

    def test_consistency_loss_nonnegative(self):
        """KL divergence should be non-negative."""
        loss_fn = MERALoss(label_smoothing=0.0)
        outputs = {
            "final_logits": torch.randn(16, 4),
            "text_logits": torch.randn(16, 4),
            "audio_logits": torch.randn(16, 4),
        }
        targets = torch.randint(0, 4, (16,))
        result = loss_fn(outputs, targets)
        assert result["consistency"].item() >= 0.0

    def test_class_weights(self):
        """MERALoss should accept class_weights tensor."""
        weights = torch.tensor([2.0, 1.0, 1.5, 0.5])
        loss_fn = MERALoss(class_weights=weights, label_smoothing=0.0)
        outputs = {
            "final_logits": torch.randn(8, 4),
            "text_logits": torch.randn(8, 4),
            "audio_logits": torch.randn(8, 4),
        }
        targets = torch.randint(0, 4, (8,))
        result = loss_fn(outputs, targets)
        assert result["total"].item() > 0

    def test_gradient_flows(self):
        """Gradients should flow through the total loss to final_logits."""
        loss_fn = MERALoss(label_smoothing=0.0)
        final_logits = torch.randn(4, 4, requires_grad=True)
        outputs = {
            "final_logits": final_logits,
            "text_logits": torch.randn(4, 4, requires_grad=True),
            "audio_logits": torch.randn(4, 4, requires_grad=True),
        }
        targets = torch.tensor([0, 1, 2, 3])
        result = loss_fn(outputs, targets)
        result["total"].backward()
        assert final_logits.grad is not None
        assert not torch.all(final_logits.grad == 0)


# ============================================================================
# Tests: PipelineLoss
# ============================================================================

class TestPipelineLoss:
    """Tests for the simple pipeline cross-entropy loss."""

    def test_output_keys(self):
        loss_fn = PipelineLoss(label_smoothing=0.0)
        logits = torch.randn(8, 4)
        targets = torch.randint(0, 4, (8,))
        result = loss_fn(logits, targets)
        assert "total" in result
        assert "ce" in result

    def test_total_equals_ce(self):
        loss_fn = PipelineLoss(label_smoothing=0.0)
        logits = torch.randn(8, 4)
        targets = torch.randint(0, 4, (8,))
        result = loss_fn(logits, targets)
        torch.testing.assert_close(result["total"], result["ce"])

    def test_perfect_prediction(self):
        loss_fn = PipelineLoss(label_smoothing=0.0)
        targets = torch.tensor([0, 1, 2, 3])
        logits = torch.full((4, 4), -10.0)
        for i in range(4):
            logits[i, i] = 10.0
        result = loss_fn(logits, targets)
        assert result["total"].item() < 0.01

    def test_with_class_weights(self):
        weights = torch.tensor([1.0, 1.0, 1.0, 1.0])
        loss_fn = PipelineLoss(class_weights=weights, label_smoothing=0.0)
        logits = torch.randn(8, 4)
        targets = torch.randint(0, 4, (8,))
        result = loss_fn(logits, targets)
        assert result["total"].item() > 0


# ============================================================================
# Tests: compute_class_weights
# ============================================================================

class TestClassWeights:
    """Tests for compute_class_weights."""

    def test_inverse_freq_balanced(self):
        """Balanced classes should produce equal weights."""
        labels = [0, 1, 2, 3] * 25  # 25 each
        weights = compute_class_weights(labels, num_classes=4, method="inverse_freq")
        assert weights.shape == (4,)
        # All should be approximately equal
        assert torch.allclose(weights, weights[0].expand(4), atol=1e-4)

    def test_inverse_freq_imbalanced(self):
        """Rare class should get higher weight."""
        labels = [0] * 100 + [1] * 10 + [2] * 50 + [3] * 40
        weights = compute_class_weights(labels, num_classes=4, method="inverse_freq")
        # Class 1 (10 samples) should have highest weight
        assert weights[1] == weights.max()
        # Class 0 (100 samples) should have lowest weight
        assert weights[0] == weights.min()

    def test_effective_num(self):
        """effective_num method should also produce valid weights."""
        labels = [0] * 100 + [1] * 10 + [2] * 50 + [3] * 40
        weights = compute_class_weights(labels, num_classes=4, method="effective_num")
        assert weights.shape == (4,)
        assert torch.all(weights > 0)

    def test_weights_sum_to_num_classes(self):
        """Weights should be normalized to sum to num_classes."""
        labels = [0] * 30 + [1] * 70 + [2] * 50 + [3] * 50
        weights = compute_class_weights(labels, num_classes=4, method="inverse_freq")
        assert abs(weights.sum().item() - 4.0) < 1e-4

    def test_tensor_input(self):
        """Should accept a torch tensor as input."""
        labels = torch.tensor([0, 1, 2, 3, 0, 1])
        weights = compute_class_weights(labels, num_classes=4)
        assert weights.shape == (4,)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            compute_class_weights([0, 1, 2], method="invalid")


# ============================================================================
# Tests: MetricTracker
# ============================================================================

class TestMetricTracker:
    """Tests for the MetricTracker class."""

    def test_perfect_predictions(self):
        """Perfect predictions should give WA=1, UA=1, F1=1."""
        tracker = MetricTracker(num_classes=4)
        # Logits where argmax == target for all samples
        logits = torch.eye(4).repeat(5, 1)  # 20 samples, 5 per class
        targets = torch.tensor([0, 1, 2, 3] * 5)
        tracker.update(logits, targets)
        results = tracker.compute()

        assert results["wa"] == pytest.approx(1.0)
        assert results["ua"] == pytest.approx(1.0)
        assert results["macro_f1"] == pytest.approx(1.0)
        assert results["total_samples"] == 20

    def test_all_wrong_predictions(self):
        """All wrong: predict class (c+1)%4 for every sample of class c."""
        tracker = MetricTracker(num_classes=4)
        # Logits shifted by 1
        logits = torch.zeros(20, 4)
        targets = torch.tensor([0, 1, 2, 3] * 5)
        for i, t in enumerate(targets):
            logits[i, (t.item() + 1) % 4] = 10.0
        tracker.update(logits, targets)
        results = tracker.compute()

        assert results["wa"] == pytest.approx(0.0)
        assert results["ua"] == pytest.approx(0.0)

    def test_partial_accuracy(self):
        """Half correct, half wrong."""
        tracker = MetricTracker(num_classes=2)
        # 4 samples: 2 correct, 2 wrong
        logits = torch.tensor([
            [10.0, -10.0],  # pred=0, target=0 ✓
            [-10.0, 10.0],  # pred=1, target=1 ✓
            [10.0, -10.0],  # pred=0, target=1 ✗
            [-10.0, 10.0],  # pred=1, target=0 ✗
        ])
        targets = torch.tensor([0, 1, 1, 0])
        tracker.update(logits, targets)
        results = tracker.compute()

        assert results["wa"] == pytest.approx(0.5)
        assert results["ua"] == pytest.approx(0.5)

    def test_multi_batch_accumulation(self):
        """Metrics should correctly accumulate across multiple update calls."""
        tracker = MetricTracker(num_classes=4)

        # Batch 1: 4 perfect predictions
        logits1 = torch.eye(4) * 10
        targets1 = torch.tensor([0, 1, 2, 3])
        tracker.update(logits1, targets1)

        # Batch 2: 4 perfect predictions
        logits2 = torch.eye(4) * 10
        targets2 = torch.tensor([0, 1, 2, 3])
        tracker.update(logits2, targets2)

        results = tracker.compute()
        assert results["wa"] == pytest.approx(1.0)
        assert results["total_samples"] == 8

    def test_confusion_matrix(self):
        """Confusion matrix should match known predictions."""
        tracker = MetricTracker(num_classes=2)
        # 2 samples: both predict class 0
        logits = torch.tensor([[10.0, -10.0], [10.0, -10.0]])
        targets = torch.tensor([0, 1])  # One correct (TP for 0), one wrong (FN for 1)
        tracker.update(logits, targets)
        results = tracker.compute()

        cm = results["confusion_matrix"]
        assert cm[0, 0] == 1  # True class 0, predicted 0
        assert cm[1, 0] == 1  # True class 1, predicted 0
        assert cm[0, 1] == 0
        assert cm[1, 1] == 0

    def test_reset(self):
        """After reset, tracker should return empty results."""
        tracker = MetricTracker(num_classes=4)
        tracker.update(torch.eye(4), torch.tensor([0, 1, 2, 3]))
        tracker.reset()
        results = tracker.compute()
        assert results["total_samples"] == 0
        assert results["wa"] == 0.0

    def test_empty_tracker(self):
        """Empty tracker should return zeros."""
        tracker = MetricTracker(num_classes=4)
        results = tracker.compute()
        assert results["total_samples"] == 0
        assert results["wa"] == 0.0
        assert results["ua"] == 0.0

    def test_per_class_f1(self):
        """Per-class F1 for a known scenario."""
        tracker = MetricTracker(num_classes=2, label_names=["cat", "dog"])
        # Perfect on class 0, wrong on class 1
        logits = torch.tensor([
            [10.0, -10.0],  # pred=0, tgt=0 ✓
            [10.0, -10.0],  # pred=0, tgt=0 ✓
            [10.0, -10.0],  # pred=0, tgt=1 ✗
            [10.0, -10.0],  # pred=0, tgt=1 ✗
        ])
        targets = torch.tensor([0, 0, 1, 1])
        tracker.update(logits, targets)
        results = tracker.compute()

        # Class 0: TP=2, FP=2, FN=0 → precision=0.5, recall=1.0, F1=2/3
        assert results["per_class_f1"]["cat"] == pytest.approx(2.0 / 3.0, abs=1e-6)
        # Class 1: TP=0, FP=0, FN=2 → F1=0
        assert results["per_class_f1"]["dog"] == pytest.approx(0.0)


# ============================================================================
# Tests: Standalone metric functions
# ============================================================================

class TestStandaloneMetrics:
    """Tests for the standalone metric computation functions."""

    def test_weighted_accuracy(self):
        preds = np.array([0, 1, 2, 3, 0])
        targets = np.array([0, 1, 2, 3, 1])
        assert compute_weighted_accuracy(preds, targets) == pytest.approx(0.8)

    def test_unweighted_accuracy_balanced(self):
        preds = np.array([0, 1, 2, 3])
        targets = np.array([0, 1, 2, 3])
        ua, _ = compute_unweighted_accuracy(preds, targets, num_classes=4)
        assert ua == pytest.approx(1.0)

    def test_unweighted_accuracy_imbalanced(self):
        """UA should handle imbalanced classes differently than WA."""
        # 8 samples of class 0 (all correct), 2 samples of class 1 (all wrong)
        preds = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        targets = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        wa = compute_weighted_accuracy(preds, targets)
        ua, per_class = compute_unweighted_accuracy(preds, targets, num_classes=2)
        assert wa == pytest.approx(0.8)
        assert ua == pytest.approx(0.5)  # (1.0 + 0.0) / 2
        assert per_class[0] == pytest.approx(1.0)
        assert per_class[1] == pytest.approx(0.0)

    def test_f1_scores(self):
        preds = np.array([0, 0, 1, 1])
        targets = np.array([0, 1, 0, 1])
        per_class_f1, macro_f1 = compute_f1_scores(preds, targets, num_classes=2)
        # For each class: TP=1, FP=1, FN=1 → precision=0.5, recall=0.5, F1=0.5
        assert per_class_f1[0] == pytest.approx(0.5)
        assert per_class_f1[1] == pytest.approx(0.5)
        assert macro_f1 == pytest.approx(0.5)

    def test_confusion_matrix(self):
        preds = np.array([0, 0, 1, 1])
        targets = np.array([0, 1, 0, 1])
        cm = compute_confusion_matrix(preds, targets, num_classes=2)
        expected = np.array([[1, 1], [1, 1]])
        np.testing.assert_array_equal(cm, expected)


# ============================================================================
# Tests: Scheduler
# ============================================================================

class TestScheduler:
    """Tests for the cosine warmup scheduler."""

    def test_warmup_starts_near_zero(self):
        """LR should start near 0 during warmup phase."""
        model = nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = build_cosine_warmup_scheduler(optimizer, total_steps=100, warmup_ratio=0.1)

        # At step 0, LR multiplier should be 0
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-8)

    def test_warmup_ramp(self):
        """LR should increase linearly during warmup."""
        model = nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
        scheduler = build_cosine_warmup_scheduler(
            optimizer, total_steps=100, warmup_ratio=0.1
        )

        lrs = []
        for step in range(10):  # warmup_steps = 10
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

        # LR should be monotonically increasing during warmup
        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i - 1], f"LR decreased during warmup: {lrs}"

    def test_peak_at_warmup_end(self):
        """LR should reach peak value at end of warmup."""
        base_lr = 1e-3
        model = nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
        scheduler = build_cosine_warmup_scheduler(
            optimizer, total_steps=100, warmup_ratio=0.1
        )

        # Step through warmup (10 steps)
        for _ in range(10):
            optimizer.step()
            scheduler.step()

        # After warmup, LR should be approximately base_lr
        lr = optimizer.param_groups[0]["lr"]
        assert lr == pytest.approx(base_lr, rel=0.05)

    def test_cosine_decay(self):
        """After warmup, LR should decay following cosine curve."""
        model = nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
        scheduler = build_cosine_warmup_scheduler(
            optimizer, total_steps=100, warmup_ratio=0.1, min_lr_ratio=0.01
        )

        # Step through all 100 steps
        lrs = []
        for _ in range(100):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

        # After warmup (step 10), LR should generally decrease
        post_warmup_lrs = lrs[10:]
        # The final LR should be near min_lr_ratio
        assert lrs[-1] == pytest.approx(0.01, abs=0.02)

    def test_min_lr_at_end(self):
        """At the end of training, LR should be approximately min_lr_ratio * base_lr."""
        base_lr = 1e-3
        min_lr_ratio = 0.01
        model = nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
        scheduler = build_cosine_warmup_scheduler(
            optimizer, total_steps=200, warmup_ratio=0.1, min_lr_ratio=min_lr_ratio
        )

        for _ in range(200):
            optimizer.step()
            scheduler.step()

        expected_min = base_lr * min_lr_ratio
        actual = optimizer.param_groups[0]["lr"]
        assert actual == pytest.approx(expected_min, rel=0.1)


# ============================================================================
# Tests: EarlyStopping
# ============================================================================

class TestEarlyStopping:
    """Tests for the EarlyStopping class."""

    def test_first_call_never_stops(self):
        """First call should never trigger stopping."""
        es = EarlyStopping(patience=3, mode="max")
        assert es(score=0.5, epoch=1) is False
        assert es.best_score == 0.5

    def test_improvement_resets_counter(self):
        """Improvements should reset the counter."""
        es = EarlyStopping(patience=3, mode="max")
        es(score=0.5, epoch=1)
        es(score=0.4, epoch=2)  # no improvement
        assert es.counter == 1
        es(score=0.6, epoch=3)  # improvement!
        assert es.counter == 0
        assert es.best_score == 0.6

    def test_patience_triggers_stop(self):
        """After `patience` epochs without improvement, should_stop = True."""
        es = EarlyStopping(patience=3, mode="max")
        es(score=0.8, epoch=1)
        es(score=0.7, epoch=2)
        es(score=0.6, epoch=3)
        result = es(score=0.5, epoch=4)  # 3rd non-improvement → stop
        assert result is True
        assert es.should_stop is True

    def test_min_mode(self):
        """In 'min' mode, lower scores are improvements."""
        es = EarlyStopping(patience=2, mode="min")
        es(score=1.0, epoch=1)
        assert es.best_score == 1.0
        es(score=0.8, epoch=2)  # improvement (lower)
        assert es.best_score == 0.8
        assert es.counter == 0
        es(score=0.9, epoch=3)  # no improvement (higher)
        assert es.counter == 1

    def test_min_delta(self):
        """Improvements smaller than min_delta should not count."""
        es = EarlyStopping(patience=3, mode="max", min_delta=0.01)
        es(score=0.80, epoch=1)
        es(score=0.805, epoch=2)  # +0.005 < min_delta → not improvement
        assert es.counter == 1
        es(score=0.815, epoch=3)  # +0.015 > min_delta → improvement
        assert es.counter == 0

    def test_checkpoint_saving(self):
        """When a checkpoint_path is given, best model should be saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "best.pt"
            es = EarlyStopping(patience=3, mode="max", checkpoint_path=ckpt_path)
            model = nn.Linear(4, 4)
            es(score=0.5, epoch=1, model=model)
            assert ckpt_path.exists()

    def test_state_dict_roundtrip(self):
        """state_dict and load_state_dict should preserve state."""
        es1 = EarlyStopping(patience=5, mode="max")
        es1(score=0.7, epoch=1)
        es1(score=0.6, epoch=2)
        es1(score=0.5, epoch=3)

        state = es1.state_dict()
        es2 = EarlyStopping(patience=5, mode="max")
        es2.load_state_dict(state)

        assert es2.best_score == es1.best_score
        assert es2.best_epoch == es1.best_epoch
        assert es2.counter == es1.counter
        assert es2.should_stop == es1.should_stop


# ============================================================================
# Tests: Trainer (one-batch smoke test)
# ============================================================================

class SimpleModel(nn.Module):
    """Tiny model for trainer smoke tests."""

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


class TestTrainer:
    """Smoke tests for the Trainer class."""

    def _make_loader(self, n_samples=32, input_dim=8, num_classes=4, batch_size=8):
        X = torch.randn(n_samples, input_dim)
        y = torch.randint(0, num_classes, (n_samples,))
        ds = TensorDataset(X, y)
        return DataLoader(ds, batch_size=batch_size, shuffle=True)

    def test_one_epoch_no_error(self, device):
        """Trainer should run one epoch without errors."""
        model = SimpleModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = PipelineLoss(label_smoothing=0.0)
        simple_forward_fn._loss_fn = loss_fn

        train_loader = self._make_loader()
        val_loader = self._make_loader(n_samples=16)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            total_epochs=1,
            forward_fn=simple_forward_fn,
            num_classes=4,
        )
        results = trainer.train()

        assert "best_metric" in results
        assert "train_history" in results
        assert len(results["train_history"]) == 1

    def test_gradient_accumulation(self, device):
        """Trainer should work with gradient accumulation > 1."""
        model = SimpleModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = PipelineLoss(label_smoothing=0.0)
        simple_forward_fn._loss_fn = loss_fn

        train_loader = self._make_loader(n_samples=32, batch_size=8)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            train_loader=train_loader,
            total_epochs=1,
            gradient_accumulation_steps=2,
            forward_fn=simple_forward_fn,
            num_classes=4,
        )
        results = trainer.train()
        assert len(results["train_history"]) == 1

    def test_with_scheduler(self, device):
        """Trainer should work with a scheduler."""
        model = SimpleModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = PipelineLoss(label_smoothing=0.0)
        simple_forward_fn._loss_fn = loss_fn

        train_loader = self._make_loader()
        total_steps = len(train_loader) * 2
        scheduler = build_cosine_warmup_scheduler(optimizer, total_steps)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            train_loader=train_loader,
            total_epochs=2,
            scheduler=scheduler,
            forward_fn=simple_forward_fn,
            num_classes=4,
        )
        results = trainer.train()
        assert len(results["train_history"]) == 2

    def test_early_stopping_integration(self, device):
        """Trainer should stop early when EarlyStopping triggers."""
        model = SimpleModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = PipelineLoss(label_smoothing=0.0)
        simple_forward_fn._loss_fn = loss_fn

        train_loader = self._make_loader()
        val_loader = self._make_loader(n_samples=16)

        es = EarlyStopping(patience=2, mode="max")

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            total_epochs=100,  # Would take forever without early stopping
            early_stopping=es,
            forward_fn=simple_forward_fn,
            num_classes=4,
        )
        results = trainer.train()
        # With random model and patience=2, should stop well before 100 epochs
        assert len(results["train_history"]) < 100

    def test_checkpoint_saving(self, device):
        """Trainer should save epoch checkpoints when checkpoint_dir is set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            loss_fn = PipelineLoss(label_smoothing=0.0)
            simple_forward_fn._loss_fn = loss_fn

            train_loader = self._make_loader()

            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                train_loader=train_loader,
                total_epochs=2,
                forward_fn=simple_forward_fn,
                num_classes=4,
                checkpoint_dir=tmpdir,
            )
            trainer.train()

            # Should have epoch checkpoints
            ckpts = list(Path(tmpdir).glob("epoch_*.pt"))
            assert len(ckpts) == 2

    def test_val_metrics_populated(self, device):
        """Val metrics should be populated when val_loader is given."""
        model = SimpleModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = PipelineLoss(label_smoothing=0.0)
        simple_forward_fn._loss_fn = loss_fn

        train_loader = self._make_loader()
        val_loader = self._make_loader(n_samples=16)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            total_epochs=1,
            forward_fn=simple_forward_fn,
            num_classes=4,
        )
        results = trainer.train()

        assert len(results["val_history"]) == 1
        assert "ua" in results["val_history"][0]
        assert "wa" in results["val_history"][0]
        assert "loss" in results["val_history"][0]
