#!/usr/bin/env python3
"""Tests for the MERA inference pipeline.

Covers:
- MERAInference instantiation (with mock checkpoint)
- _build_result produces all expected output fields
- _generate_interpretation generates correct text for different scenarios
- predict_from_features works with synthetic tensors
- Missing audio file raises FileNotFoundError
- Result dict has correct types and shapes
- Works on both MPS and CPU devices
- JSON output serialisation
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_inference import MERAInference, EMOTION_LABELS


# ============================================================================
# Fixtures & Helpers
# ============================================================================

@pytest.fixture
def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MockMERAModel(nn.Module):
    """Lightweight mock of MERAModel for testing without DeBERTa download."""

    def __init__(self, embed_dim=256, num_classes=4, num_layers=12):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, **kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        device = kwargs["input_ids"].device

        # Generate deterministic outputs
        final_logits = torch.randn(batch_size, self.num_classes, device=device)
        text_logits = torch.randn(batch_size, self.num_classes, device=device)
        audio_logits = torch.randn(batch_size, self.num_classes, device=device)
        gate_weights = F.softmax(torch.randn(batch_size, 2, device=device), dim=1)
        attention_entropy = torch.abs(torch.randn(batch_size, self.num_layers, device=device))

        return {
            "final_logits": final_logits,
            "text_logits": text_logits,
            "audio_logits": audio_logits,
            "gate_weights": gate_weights,
            "attention_entropy": attention_entropy,
        }


class MockTokenizer:
    """Mock tokenizer that returns dummy tensors."""

    def __call__(self, text, **kwargs):
        if isinstance(text, str):
            text = [text]
        batch_size = len(text)
        seq_len = 10
        return {
            "input_ids": torch.ones(batch_size, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        }


def make_mock_pipeline(device):
    """Create a MockMERAModel-based inference pipeline without loading DeBERTa."""
    pipeline = object.__new__(MERAInference)
    pipeline.model = MockMERAModel().float().to(device)
    pipeline.model.eval()
    pipeline.device = device
    pipeline._tokenizer = MockTokenizer()
    pipeline._whisper_model_id = "mock"

    # Mock the config
    class MockConfig:
        class data:
            sample_rate = 16000
            max_audio_duration_sec = 15.0
        class pipeline_a:
            max_text_length = 128
        class fusion:
            embed_dim = 256

    pipeline.config = MockConfig()
    return pipeline


# ============================================================================
# Tests: _build_result
# ============================================================================

class TestBuildResult:
    """Tests for the _build_result method."""

    def test_all_expected_keys(self, device):
        """Result dict should contain all expected keys."""
        pipeline = make_mock_pipeline(device)

        # Create mock model outputs
        outputs = {
            "final_logits": torch.tensor([[2.0, 0.5, -1.0, 0.3]], device=device),
            "text_logits": torch.tensor([[1.5, 0.8, -0.5, 0.2]], device=device),
            "audio_logits": torch.tensor([[1.0, 0.3, 0.5, -0.2]], device=device),
            "gate_weights": torch.tensor([[0.4, 0.6]], device=device),
            "attention_entropy": torch.ones(1, 12, device=device) * 1.5,
        }

        result = pipeline._build_result(outputs, "test transcript")

        expected_keys = {
            "emotion", "confidence", "probabilities", "transcript",
            "gate_weights", "attention_entropy", "attention_entropy_mean",
            "interpretation", "text_prediction", "audio_prediction",
            "text_probabilities", "audio_probabilities",
        }
        assert set(result.keys()) == expected_keys

    def test_emotion_is_valid_label(self, device):
        """Predicted emotion should be one of the 4 valid labels."""
        pipeline = make_mock_pipeline(device)
        outputs = {
            "final_logits": torch.tensor([[5.0, -1.0, -2.0, 0.0]], device=device),
            "text_logits": torch.tensor([[3.0, 0.0, 0.0, 0.0]], device=device),
            "audio_logits": torch.tensor([[2.0, 0.0, 0.0, 0.0]], device=device),
            "gate_weights": torch.tensor([[0.5, 0.5]], device=device),
            "attention_entropy": torch.ones(1, 12, device=device),
        }
        result = pipeline._build_result(outputs, "hello")

        assert result["emotion"] in EMOTION_LABELS
        # With logits [5, -1, -2, 0], should predict "angry" (index 0)
        assert result["emotion"] == "angry"

    def test_confidence_in_range(self, device):
        """Confidence should be between 0 and 1."""
        pipeline = make_mock_pipeline(device)
        outputs = {
            "final_logits": torch.randn(1, 4, device=device),
            "text_logits": torch.randn(1, 4, device=device),
            "audio_logits": torch.randn(1, 4, device=device),
            "gate_weights": torch.tensor([[0.5, 0.5]], device=device),
            "attention_entropy": torch.ones(1, 12, device=device),
        }
        result = pipeline._build_result(outputs, "test")

        assert 0.0 <= result["confidence"] <= 1.0

    def test_probabilities_sum_to_one(self, device):
        """Probabilities should sum to approximately 1."""
        pipeline = make_mock_pipeline(device)
        outputs = {
            "final_logits": torch.randn(1, 4, device=device),
            "text_logits": torch.randn(1, 4, device=device),
            "audio_logits": torch.randn(1, 4, device=device),
            "gate_weights": torch.tensor([[0.5, 0.5]], device=device),
            "attention_entropy": torch.ones(1, 12, device=device),
        }
        result = pipeline._build_result(outputs, "test")

        prob_sum = sum(result["probabilities"].values())
        assert prob_sum == pytest.approx(1.0, abs=1e-3)

    def test_gate_weights_structure(self, device):
        """Gate weights should have 'text' and 'audio' keys."""
        pipeline = make_mock_pipeline(device)
        outputs = {
            "final_logits": torch.randn(1, 4, device=device),
            "text_logits": torch.randn(1, 4, device=device),
            "audio_logits": torch.randn(1, 4, device=device),
            "gate_weights": torch.tensor([[0.35, 0.65]], device=device),
            "attention_entropy": torch.ones(1, 12, device=device),
        }
        result = pipeline._build_result(outputs, "test")

        assert "text" in result["gate_weights"]
        assert "audio" in result["gate_weights"]
        assert result["gate_weights"]["text"] == pytest.approx(0.35, abs=1e-3)
        assert result["gate_weights"]["audio"] == pytest.approx(0.65, abs=1e-3)

    def test_attention_entropy_shape(self, device):
        """Attention entropy should be a list of 12 floats."""
        pipeline = make_mock_pipeline(device)
        outputs = {
            "final_logits": torch.randn(1, 4, device=device),
            "text_logits": torch.randn(1, 4, device=device),
            "audio_logits": torch.randn(1, 4, device=device),
            "gate_weights": torch.tensor([[0.5, 0.5]], device=device),
            "attention_entropy": torch.ones(1, 12, device=device) * 1.5,
        }
        result = pipeline._build_result(outputs, "test")

        assert len(result["attention_entropy"]) == 12
        assert all(isinstance(e, float) for e in result["attention_entropy"])
        assert result["attention_entropy_mean"] == pytest.approx(1.5, abs=1e-3)

    def test_transcript_preserved(self, device):
        """Transcript should be preserved in the result."""
        pipeline = make_mock_pipeline(device)
        outputs = {
            "final_logits": torch.randn(1, 4, device=device),
            "text_logits": torch.randn(1, 4, device=device),
            "audio_logits": torch.randn(1, 4, device=device),
            "gate_weights": torch.tensor([[0.5, 0.5]], device=device),
            "attention_entropy": torch.ones(1, 12, device=device),
        }
        result = pipeline._build_result(outputs, "I am very angry right now")
        assert result["transcript"] == "I am very angry right now"


# ============================================================================
# Tests: _generate_interpretation
# ============================================================================

class TestInterpretation:
    """Tests for the interpretation string generator."""

    def test_high_confidence_text(self):
        """High confidence should mention 'High confidence'."""
        interp = MERAInference._generate_interpretation(
            "angry", 0.92, {"text": 0.5, "audio": 0.5}, 1.5,
        )
        assert "High confidence" in interp

    def test_low_confidence_text(self):
        """Low confidence should mention 'Low confidence'."""
        interp = MERAInference._generate_interpretation(
            "happy", 0.35, {"text": 0.5, "audio": 0.5}, 1.5,
        )
        assert "Low confidence" in interp

    def test_text_dominant_interpretation(self):
        """When text weight >> audio, should mention text reliance."""
        interp = MERAInference._generate_interpretation(
            "sad", 0.7, {"text": 0.8, "audio": 0.2}, 1.5,
        )
        assert "text" in interp.lower()

    def test_audio_dominant_interpretation(self):
        """When audio weight >> text, should mention vocal cues."""
        interp = MERAInference._generate_interpretation(
            "neutral", 0.7, {"text": 0.2, "audio": 0.8}, 1.5,
        )
        assert "vocal" in interp.lower()

    def test_balanced_interpretation(self):
        """When weights are balanced, should mention both."""
        interp = MERAInference._generate_interpretation(
            "angry", 0.7, {"text": 0.48, "audio": 0.52}, 1.5,
        )
        assert "both" in interp.lower() or "equally" in interp.lower()

    def test_high_entropy_interpretation(self):
        """High entropy should mention ambiguous text."""
        interp = MERAInference._generate_interpretation(
            "angry", 0.7, {"text": 0.5, "audio": 0.5}, 2.5,
        )
        assert "ambiguous" in interp.lower()

    def test_low_entropy_interpretation(self):
        """Low entropy should mention clear text."""
        interp = MERAInference._generate_interpretation(
            "angry", 0.7, {"text": 0.5, "audio": 0.5}, 0.5,
        )
        assert "clear" in interp.lower()


# ============================================================================
# Tests: predict_from_features
# ============================================================================

class TestPredictFromFeatures:
    """Tests for predict_from_features with synthetic tensors."""

    def test_returns_result_dict(self, device):
        """predict_from_features should return a valid result dict."""
        pipeline = make_mock_pipeline(device)

        spec = torch.randn(128, 100)
        prosody = torch.randn(10, 100)
        hubert = torch.randn(25, 50, 1024)

        result = pipeline.predict_from_features(
            transcript="I am so happy",
            spectrogram=spec,
            prosody=prosody,
            hubert=hubert,
        )

        assert "emotion" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert result["emotion"] in EMOTION_LABELS

    def test_different_sequence_lengths(self, device):
        """Should handle different time dimensions in features."""
        pipeline = make_mock_pipeline(device)

        for spec_t, pros_t, hub_t in [(50, 50, 25), (200, 200, 100), (10, 10, 5)]:
            spec = torch.randn(128, spec_t)
            prosody = torch.randn(10, pros_t)
            hubert = torch.randn(25, hub_t, 1024)

            result = pipeline.predict_from_features(
                transcript="test",
                spectrogram=spec,
                prosody=prosody,
                hubert=hubert,
            )
            assert result["emotion"] in EMOTION_LABELS


# ============================================================================
# Tests: Error handling
# ============================================================================

class TestErrorHandling:
    """Tests for error handling in the inference pipeline."""

    def test_missing_audio_file(self, device):
        """predict() should raise FileNotFoundError for missing file."""
        pipeline = make_mock_pipeline(device)

        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            pipeline.predict("/nonexistent/path/audio.wav")


# ============================================================================
# Tests: JSON serialisation
# ============================================================================

class TestJSONSerialisation:
    """Tests that inference results are JSON-serialisable."""

    def test_result_is_json_serialisable(self, device):
        """The result dict should be fully JSON-serialisable."""
        pipeline = make_mock_pipeline(device)

        outputs = {
            "final_logits": torch.randn(1, 4, device=device),
            "text_logits": torch.randn(1, 4, device=device),
            "audio_logits": torch.randn(1, 4, device=device),
            "gate_weights": torch.tensor([[0.5, 0.5]], device=device),
            "attention_entropy": torch.ones(1, 12, device=device),
        }
        result = pipeline._build_result(outputs, "test")

        # Should not raise
        json_str = json.dumps(result, indent=2)
        assert isinstance(json_str, str)

        # Should roundtrip
        loaded = json.loads(json_str)
        assert loaded["emotion"] == result["emotion"]
        assert loaded["confidence"] == pytest.approx(result["confidence"], abs=1e-6)

    def test_result_save_to_file(self, device):
        """Should be able to save result to a JSON file."""
        pipeline = make_mock_pipeline(device)

        outputs = {
            "final_logits": torch.tensor([[3.0, -1.0, 0.0, 0.5]], device=device),
            "text_logits": torch.randn(1, 4, device=device),
            "audio_logits": torch.randn(1, 4, device=device),
            "gate_weights": torch.tensor([[0.4, 0.6]], device=device),
            "attention_entropy": torch.ones(1, 12, device=device) * 1.2,
        }
        result = pipeline._build_result(outputs, "I'm furious")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(result, f, indent=2)
            f.flush()

            loaded = json.load(open(f.name))
            assert loaded["emotion"] == "angry"
            assert loaded["transcript"] == "I'm furious"


# ============================================================================
# Tests: Device compatibility
# ============================================================================

class TestDeviceCompatibility:
    """Tests that inference works on different devices."""

    def test_cpu_inference(self):
        """Inference should work on CPU."""
        cpu_device = torch.device("cpu")
        pipeline = make_mock_pipeline(cpu_device)

        spec = torch.randn(128, 100)
        prosody = torch.randn(10, 100)
        hubert = torch.randn(25, 50, 1024)

        result = pipeline.predict_from_features("test", spec, prosody, hubert)
        assert result["emotion"] in EMOTION_LABELS

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available",
    )
    def test_mps_inference(self):
        """Inference should work on MPS."""
        mps_device = torch.device("mps")
        pipeline = make_mock_pipeline(mps_device)

        spec = torch.randn(128, 100)
        prosody = torch.randn(10, 100)
        hubert = torch.randn(25, 50, 1024)

        result = pipeline.predict_from_features("test", spec, prosody, hubert)
        assert result["emotion"] in EMOTION_LABELS


# ============================================================================
# Tests: Probability distribution properties
# ============================================================================

class TestProbabilityProperties:
    """Tests for mathematical properties of the output probabilities."""

    def test_all_probabilities_non_negative(self, device):
        """All probabilities should be >= 0."""
        pipeline = make_mock_pipeline(device)
        outputs = {
            "final_logits": torch.randn(1, 4, device=device),
            "text_logits": torch.randn(1, 4, device=device),
            "audio_logits": torch.randn(1, 4, device=device),
            "gate_weights": torch.tensor([[0.5, 0.5]], device=device),
            "attention_entropy": torch.ones(1, 12, device=device),
        }
        result = pipeline._build_result(outputs, "test")

        for label, prob in result["probabilities"].items():
            assert prob >= 0, f"{label} has negative probability: {prob}"

    def test_text_probabilities_sum_to_one(self, device):
        """Text sub-pipeline probabilities should sum to ~1."""
        pipeline = make_mock_pipeline(device)
        outputs = {
            "final_logits": torch.randn(1, 4, device=device),
            "text_logits": torch.randn(1, 4, device=device),
            "audio_logits": torch.randn(1, 4, device=device),
            "gate_weights": torch.tensor([[0.5, 0.5]], device=device),
            "attention_entropy": torch.ones(1, 12, device=device),
        }
        result = pipeline._build_result(outputs, "test")

        text_sum = sum(result["text_probabilities"].values())
        assert text_sum == pytest.approx(1.0, abs=1e-3)

    def test_audio_probabilities_sum_to_one(self, device):
        """Audio sub-pipeline probabilities should sum to ~1."""
        pipeline = make_mock_pipeline(device)
        outputs = {
            "final_logits": torch.randn(1, 4, device=device),
            "text_logits": torch.randn(1, 4, device=device),
            "audio_logits": torch.randn(1, 4, device=device),
            "gate_weights": torch.tensor([[0.5, 0.5]], device=device),
            "attention_entropy": torch.ones(1, 12, device=device),
        }
        result = pipeline._build_result(outputs, "test")

        audio_sum = sum(result["audio_probabilities"].values())
        assert audio_sum == pytest.approx(1.0, abs=1e-3)

    def test_confidence_equals_max_probability(self, device):
        """Confidence should equal the max probability."""
        pipeline = make_mock_pipeline(device)
        outputs = {
            "final_logits": torch.tensor([[5.0, -1.0, 0.0, 0.5]], device=device),
            "text_logits": torch.randn(1, 4, device=device),
            "audio_logits": torch.randn(1, 4, device=device),
            "gate_weights": torch.tensor([[0.5, 0.5]], device=device),
            "attention_entropy": torch.ones(1, 12, device=device),
        }
        result = pipeline._build_result(outputs, "test")

        max_prob = max(result["probabilities"].values())
        assert result["confidence"] == pytest.approx(max_prob, abs=1e-3)
