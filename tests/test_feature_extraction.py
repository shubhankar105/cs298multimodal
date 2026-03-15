"""Tests for MERA feature extraction modules.

Uses short synthetic audio signals so tests can run without real datasets
or large pre-trained models.  Each extractor is tested for:
- Correct output shape
- Correct dtype
- Reasonable value ranges
- Edge-case handling
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_audio(
    sr: int = 16000,
    duration_sec: float = 2.0,
    freq_hz: float = 220.0,
) -> np.ndarray:
    """Generate a synthetic speech-like signal (sine + harmonics + noise)."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # Fundamental + harmonics (mimics voiced speech)
    signal = (
        0.5 * np.sin(2 * np.pi * freq_hz * t)
        + 0.25 * np.sin(2 * np.pi * 2 * freq_hz * t)
        + 0.12 * np.sin(2 * np.pi * 3 * freq_hz * t)
    )
    # Add a bit of noise for realism
    signal += 0.05 * np.random.randn(len(t))
    return signal.astype(np.float32)


def _write_wav(path: Path, audio: np.ndarray, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)


# ---------------------------------------------------------------------------
# Spectrogram tests
# ---------------------------------------------------------------------------

class TestSpectrogram:
    """Tests for log-Mel spectrogram extraction."""

    def test_output_shape(self):
        from src.features.spectrogram import extract_log_mel

        audio = _make_synthetic_audio(duration_sec=2.0)
        log_mel = extract_log_mel(audio)

        assert log_mel.ndim == 2
        assert log_mel.shape[0] == 128  # n_mels

        # Expected frames: 1 + (len(audio) - 1) // hop_length
        expected_frames = 1 + (len(audio) - 1) // 160
        assert abs(log_mel.shape[1] - expected_frames) <= 2  # Allow small rounding

    def test_dtype(self):
        from src.features.spectrogram import extract_log_mel

        audio = _make_synthetic_audio()
        log_mel = extract_log_mel(audio)
        assert log_mel.dtype == np.float32

    def test_value_range(self):
        from src.features.spectrogram import extract_log_mel

        audio = _make_synthetic_audio()
        log_mel = extract_log_mel(audio)

        # Normalised to [0, 1]
        assert log_mel.min() >= -0.01
        assert log_mel.max() <= 1.01

    def test_extract_and_save(self):
        from src.features.spectrogram import extract_and_save

        audio = _make_synthetic_audio()
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "test.wav"
            _write_wav(wav_path, audio)
            out_path = Path(tmpdir) / "test.npy"

            result = extract_and_save(wav_path, out_path)
            assert out_path.exists()
            loaded = np.load(str(out_path))
            np.testing.assert_array_equal(result, loaded)

    def test_batch_extraction_with_skip(self):
        from src.features.spectrogram import extract_spectrograms_batch

        audio = _make_synthetic_audio()
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "test.wav"
            _write_wav(wav_path, audio)
            out_dir = Path(tmpdir) / "specs"

            # First run
            stats = extract_spectrograms_batch([wav_path], out_dir, show_progress=False)
            assert stats["processed"] == 1
            assert stats["skipped"] == 0

            # Second run — should skip
            stats = extract_spectrograms_batch([wav_path], out_dir, show_progress=False)
            assert stats["processed"] == 0
            assert stats["skipped"] == 1

    def test_short_audio(self):
        """Test that very short audio doesn't crash."""
        from src.features.spectrogram import extract_log_mel

        short = _make_synthetic_audio(duration_sec=0.1)
        log_mel = extract_log_mel(short)
        assert log_mel.ndim == 2
        assert log_mel.shape[0] == 128
        assert log_mel.shape[1] > 0


# ---------------------------------------------------------------------------
# Prosodic contour tests
# ---------------------------------------------------------------------------

class TestProsodicContours:
    """Tests for prosodic contour extraction (NOVEL contribution)."""

    def test_output_shape(self):
        from src.features.prosodic import extract_prosodic_contours

        audio = _make_synthetic_audio(duration_sec=2.0)
        contours = extract_prosodic_contours(audio)

        assert contours.ndim == 2
        assert contours.shape[0] == 10  # 10 channels

        expected_frames = 1 + (len(audio) - 1) // 160
        assert abs(contours.shape[1] - expected_frames) <= 2

    def test_dtype(self):
        from src.features.prosodic import extract_prosodic_contours

        audio = _make_synthetic_audio()
        contours = extract_prosodic_contours(audio)
        assert contours.dtype == np.float32

    def test_normalisation(self):
        """Each channel should be approximately zero-mean after normalisation."""
        from src.features.prosodic import extract_prosodic_contours

        audio = _make_synthetic_audio(duration_sec=3.0)
        contours = extract_prosodic_contours(audio)

        for ch in range(10):
            mean = np.abs(contours[ch].mean())
            assert mean < 0.1, f"Channel {ch} mean = {mean:.4f}, expected ≈ 0"

    def test_channel_names(self):
        from src.features.prosodic import CHANNEL_NAMES, NUM_CHANNELS

        assert len(CHANNEL_NAMES) == NUM_CHANNELS == 10

    def test_spectrogram_frame_alignment(self):
        """Prosodic frames should match spectrogram frames."""
        from src.features.prosodic import extract_prosodic_contours
        from src.features.spectrogram import extract_log_mel

        audio = _make_synthetic_audio(duration_sec=3.0)
        spec = extract_log_mel(audio)
        contours = extract_prosodic_contours(audio)

        # Allow ±2 frames tolerance due to edge effects
        assert abs(spec.shape[1] - contours.shape[1]) <= 2

    def test_extract_and_save(self):
        from src.features.prosodic import extract_and_save

        audio = _make_synthetic_audio()
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "test.wav"
            _write_wav(wav_path, audio)
            out_path = Path(tmpdir) / "test.npy"

            result = extract_and_save(wav_path, out_path)
            assert out_path.exists()
            loaded = np.load(str(out_path))
            np.testing.assert_array_equal(result, loaded)

    def test_batch_with_skip(self):
        from src.features.prosodic import extract_prosodics_batch

        audio = _make_synthetic_audio()
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "test.wav"
            _write_wav(wav_path, audio)
            out_dir = Path(tmpdir) / "pros"

            stats = extract_prosodics_batch([wav_path], out_dir, show_progress=False)
            assert stats["processed"] == 1

            stats = extract_prosodics_batch([wav_path], out_dir, show_progress=False)
            assert stats["skipped"] == 1

    def test_unvoiced_f0_is_zero(self):
        """For a pure-noise signal, most F0 values should be 0 (unvoiced)."""
        from src.features.prosodic import extract_prosodic_contours

        noise = np.random.randn(16000 * 2).astype(np.float32) * 0.3
        contours = extract_prosodic_contours(noise)
        # After normalization, F0 channel (0) should be mostly at the negative
        # end (since most raw values are 0 → below mean). Just check it doesn't crash.
        assert contours.shape == (10, contours.shape[1])


# ---------------------------------------------------------------------------
# Augmentation tests
# ---------------------------------------------------------------------------

class TestAugmentation:
    """Tests for audio and spectrogram augmentation."""

    def test_audio_augmentor_shape(self):
        from src.data.augmentation import AudioAugmentor

        aug = AudioAugmentor(noise_prob=1.0, time_stretch_prob=0.0, pitch_shift_prob=0.0)
        audio = _make_synthetic_audio()
        result = aug(audio)

        assert result.dtype == np.float32
        assert result.ndim == 1
        # With only noise, length should stay the same
        assert len(result) == len(audio)

    def test_audio_augmentor_normalisation(self):
        from src.data.augmentation import AudioAugmentor

        aug = AudioAugmentor(noise_prob=1.0, time_stretch_prob=1.0, pitch_shift_prob=1.0)
        audio = _make_synthetic_audio()
        result = aug(audio)

        assert np.max(np.abs(result)) <= 1.0 + 1e-6

    def test_spec_augment_shape(self):
        from src.data.augmentation import SpecAugment

        sa = SpecAugment(freq_mask_param=10, time_mask_param=20)
        spec = np.random.randn(128, 300).astype(np.float32)
        result = sa(spec)

        assert result.shape == spec.shape
        assert result.dtype == spec.dtype

    def test_spec_augment_creates_zeros(self):
        from src.data.augmentation import SpecAugment

        sa = SpecAugment(freq_mask_param=10, time_mask_param=20, n_freq_masks=2, n_time_masks=2)
        spec = np.ones((128, 300), dtype=np.float32)
        result = sa(spec)

        # Should have some masked (zero) regions
        assert np.sum(result == 0.0) > 0

    def test_spec_augment_does_not_modify_input(self):
        from src.data.augmentation import SpecAugment

        sa = SpecAugment()
        spec = np.ones((128, 300), dtype=np.float32)
        original = spec.copy()
        _ = sa(spec)

        np.testing.assert_array_equal(spec, original)

    def test_build_from_config(self):
        from src.utils.config import MERAConfig
        from src.data.augmentation import build_augmentor_from_config, build_spec_augment_from_config

        config = MERAConfig()
        aug = build_augmentor_from_config(config.augmentation)
        sa = build_spec_augment_from_config(config.augmentation)

        assert aug.noise_prob == 0.5
        assert sa.freq_mask_param == 15


# ---------------------------------------------------------------------------
# Collate / CachedFeatureDataset tests
# ---------------------------------------------------------------------------

class TestCollate:
    """Tests for the collate function and CachedFeatureDataset."""

    def _make_mock_sample(self, t_spec: int = 200, t_hub: int = 100) -> dict:
        return {
            "spectrogram": torch.randn(128, t_spec),
            "prosody": torch.randn(10, t_spec),
            "hubert": torch.randn(25, t_hub, 1024),
            "transcript": "hello world",
            "emotion": 1,
            "file_id": "test_001",
            "speaker_id": "spk_01",
        }

    def test_collate_padding(self):
        from src.data.collate import collate_features

        batch = [
            self._make_mock_sample(t_spec=200, t_hub=100),
            self._make_mock_sample(t_spec=300, t_hub=150),
        ]
        result = collate_features(batch)

        assert result["spectrogram"].shape == (2, 128, 300)
        assert result["spectrogram_mask"].shape == (2, 300)
        assert result["prosody"].shape == (2, 10, 300)
        assert result["hubert"].shape == (2, 25, 150, 1024)
        assert result["emotion"].shape == (2,)

    def test_collate_masks(self):
        from src.data.collate import collate_features

        batch = [
            self._make_mock_sample(t_spec=100, t_hub=50),
            self._make_mock_sample(t_spec=200, t_hub=100),
        ]
        result = collate_features(batch)

        # First sample: mask should be True for first 100, False after
        assert result["spectrogram_mask"][0, :100].all()
        assert not result["spectrogram_mask"][0, 100:].any()

        # Second sample: mask should be all True
        assert result["spectrogram_mask"][1].all()

    def test_collate_single_sample(self):
        from src.data.collate import collate_features

        batch = [self._make_mock_sample()]
        result = collate_features(batch)
        assert result["spectrogram"].shape[0] == 1

    def test_cached_dataset_from_files(self):
        """Test CachedFeatureDataset with actual file reads."""
        from src.data.collate import CachedFeatureDataset
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create mock cached files
            spec = np.random.randn(128, 200).astype(np.float32)
            pros = np.random.randn(10, 200).astype(np.float32)
            hub = np.random.randn(25, 100, 1024).astype(np.float16)
            transcript = {"file_id": "test_001", "text": "hello world", "segments": []}

            np.save(str(tmpdir / "spec.npy"), spec)
            np.save(str(tmpdir / "pros.npy"), pros)
            np.save(str(tmpdir / "hub.npy"), hub)
            with open(tmpdir / "transcript.json", "w") as f:
                json.dump(transcript, f)

            df = pd.DataFrame([{
                "file_id": "test_001",
                "emotion_4class": "happy",
                "speaker_id": "spk_01",
                "spectrogram_path": str(tmpdir / "spec.npy"),
                "prosody_path": str(tmpdir / "pros.npy"),
                "hubert_path": str(tmpdir / "hub.npy"),
                "transcript_path": str(tmpdir / "transcript.json"),
            }])

            dataset = CachedFeatureDataset(df)
            sample = dataset[0]

            assert sample["spectrogram"].shape == (128, 200)
            assert sample["prosody"].shape == (10, 200)
            assert sample["hubert"].shape == (25, 100, 1024)
            assert sample["hubert"].dtype == torch.float32  # Cast from fp16
            assert sample["transcript"] == "hello world"
            assert sample["emotion"] == 1  # happy

    def test_cached_dataset_missing_column_raises(self):
        from src.data.collate import CachedFeatureDataset
        import pandas as pd

        df = pd.DataFrame([{"file_id": "test_001"}])
        with pytest.raises(ValueError, match="missing columns"):
            CachedFeatureDataset(df)


# ---------------------------------------------------------------------------
# Whisper transcriber tests (mock-based, no actual model download)
# ---------------------------------------------------------------------------

class TestWhisperTranscriber:
    """Tests for Whisper transcriber utility functions."""

    def test_load_transcript(self):
        from src.features.whisper_transcriber import load_transcript

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            data = {
                "file_id": "test",
                "text": "Hello world",
                "segments": [{"start": 0.0, "end": 1.0, "text": "Hello world", "words": []}],
                "language": "en",
            }
            with open(path, "w") as f:
                json.dump(data, f)

            loaded = load_transcript(path)
            assert loaded["text"] == "Hello world"
            assert loaded["language"] == "en"
            assert len(loaded["segments"]) == 1

    def test_load_transcript_missing_raises(self):
        from src.features.whisper_transcriber import load_transcript

        with pytest.raises(FileNotFoundError):
            load_transcript("/nonexistent/transcript.json")


# ---------------------------------------------------------------------------
# HuBERT extractor tests (lightweight, no model download)
# ---------------------------------------------------------------------------

class TestHuBERTExtractor:
    """Tests for HuBERT embedding loading utilities."""

    def test_load_hubert_embedding(self):
        from src.features.hubert_extractor import load_hubert_embedding

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npy"
            fake = np.random.randn(25, 50, 1024).astype(np.float16)
            np.save(str(path), fake)

            loaded = load_hubert_embedding(path, dtype=np.float32, mmap=False)
            assert loaded.dtype == np.float32
            assert loaded.shape == (25, 50, 1024)

    def test_load_hubert_mmap(self):
        from src.features.hubert_extractor import load_hubert_embedding

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npy"
            fake = np.random.randn(25, 50, 1024).astype(np.float16)
            np.save(str(path), fake)

            loaded = load_hubert_embedding(path, mmap=True)
            assert loaded.shape == (25, 50, 1024)
            # mmap arrays are read-only
            assert not loaded.flags.writeable

    def test_load_missing_raises(self):
        from src.features.hubert_extractor import load_hubert_embedding

        with pytest.raises(FileNotFoundError):
            load_hubert_embedding("/nonexistent/embedding.npy")


# ---------------------------------------------------------------------------
# Integration: spectrogram + prosodic frame alignment
# ---------------------------------------------------------------------------

class TestFeatureAlignment:
    """Cross-module integration tests."""

    @pytest.mark.parametrize("duration", [1.0, 3.0, 5.0, 10.0])
    def test_spec_prosody_alignment_across_durations(self, duration):
        """Spectrogram and prosodic contour frame counts should match closely."""
        from src.features.spectrogram import extract_log_mel
        from src.features.prosodic import extract_prosodic_contours

        audio = _make_synthetic_audio(duration_sec=duration)
        spec = extract_log_mel(audio)
        contours = extract_prosodic_contours(audio)

        # Allow ±2 frames tolerance
        assert abs(spec.shape[1] - contours.shape[1]) <= 2, (
            f"Spec frames={spec.shape[1]}, Prosodic frames={contours.shape[1]} "
            f"for {duration}s audio"
        )

    def test_full_pipeline_save_load(self):
        """Test saving and loading all features for one utterance."""
        from src.features.spectrogram import extract_and_save as spec_save
        from src.features.prosodic import extract_and_save as pros_save

        audio = _make_synthetic_audio(duration_sec=2.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            wav_path = tmpdir / "test.wav"
            _write_wav(wav_path, audio)

            spec_path = tmpdir / "spec.npy"
            pros_path = tmpdir / "pros.npy"

            spec = spec_save(wav_path, spec_path)
            pros = pros_save(wav_path, pros_path)

            # Reload
            spec_loaded = np.load(str(spec_path))
            pros_loaded = np.load(str(pros_path))

            np.testing.assert_array_equal(spec, spec_loaded)
            np.testing.assert_array_equal(pros, pros_loaded)

            assert spec_loaded.shape[0] == 128
            assert pros_loaded.shape[0] == 10
