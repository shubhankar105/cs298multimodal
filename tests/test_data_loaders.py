"""Tests for MERA dataset loaders.

Tests use synthetic/mock data so they can run without the actual datasets.
Each loader is tested for correct parsing, label mapping, and edge cases.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers to create synthetic dataset structures
# ---------------------------------------------------------------------------

def _write_wav(path: Path, sr: int = 16000, duration_sec: float = 1.0) -> None:
    """Write a minimal WAV file using soundfile."""
    import soundfile as sf

    samples = int(sr * duration_sec)
    audio = np.random.randn(samples).astype(np.float32) * 0.1
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)


# ---------------------------------------------------------------------------
# IEMOCAP loader tests
# ---------------------------------------------------------------------------

class TestIEMOCAPLoader:
    """Tests for the IEMOCAP dataset loader."""

    def _create_mock_session(self, session_dir: Path, session_num: int = 1) -> None:
        """Create a minimal mock IEMOCAP session directory."""
        eval_dir = session_dir / "dialog" / "EmoEvaluation"
        wav_dir = session_dir / "sentences" / "wav"
        eval_dir.mkdir(parents=True, exist_ok=True)
        wav_dir.mkdir(parents=True, exist_ok=True)

        dialog_id = f"Ses{session_num:02d}F_impro01"
        dialog_wav_dir = wav_dir / dialog_id
        dialog_wav_dir.mkdir(parents=True, exist_ok=True)

        # Create label file with various emotions
        eval_file = eval_dir / f"{dialog_id}.txt"
        lines = [
            f"[0.0 - 2.5] {dialog_id}_F000 ang [3.5, 4.0, 3.0]\n",
            f"[2.6 - 5.0] {dialog_id}_M001 hap [3.0, 3.5, 3.0]\n",
            f"[5.1 - 7.5] {dialog_id}_F002 exc [4.0, 4.0, 3.5]\n",  # Should map to happy
            f"[7.6 - 10.0] {dialog_id}_M003 sad [2.0, 2.0, 2.0]\n",
            f"[10.1 - 12.0] {dialog_id}_F004 neu [3.0, 3.0, 3.0]\n",
            f"[12.1 - 14.0] {dialog_id}_M005 fru [3.0, 3.5, 3.0]\n",  # Should be skipped
            "Some random header line\n",
        ]
        eval_file.write_text("".join(lines))

        # Create corresponding WAV files
        for suffix in ["F000", "M001", "F002", "M003", "F004", "M005"]:
            _write_wav(dialog_wav_dir / f"{dialog_id}_{suffix}.wav")

    def test_parse_session_basic(self):
        """Test basic session parsing with valid utterances."""
        from src.data.iemocap_loader import parse_iemocap_session

        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir) / "Session1"
            self._create_mock_session(session_dir, session_num=1)

            utterances = parse_iemocap_session(session_dir)

            # Should get 5 valid utterances (fru is excluded)
            assert len(utterances) == 5

    def test_emotion_mapping(self):
        """Test that emotions are correctly mapped to 4-class."""
        from src.data.iemocap_loader import parse_iemocap_session

        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir) / "Session1"
            self._create_mock_session(session_dir, session_num=1)

            utterances = parse_iemocap_session(session_dir)
            emotions = [u.emotion for u in utterances]

            assert "angry" in emotions
            assert "happy" in emotions   # Both hap and exc should map to happy
            assert "sad" in emotions
            assert "neutral" in emotions

            # Count happy — both hap and exc should map to happy
            happy_count = sum(1 for e in emotions if e == "happy")
            assert happy_count == 2

    def test_metadata_extraction(self):
        """Test that metadata (session, gender, dialog type) is correct."""
        from src.data.iemocap_loader import parse_iemocap_session

        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir) / "Session1"
            self._create_mock_session(session_dir, session_num=1)

            utterances = parse_iemocap_session(session_dir)

            for u in utterances:
                assert u.session_id == 1
                assert u.gender in ("M", "F")
                assert u.dialog_type == "improvised"  # All are impro01
                assert u.speaker_id.startswith("Ses01")

    def test_missing_session_raises(self):
        """Test that missing directory raises FileNotFoundError."""
        from src.data.iemocap_loader import parse_iemocap_session

        with pytest.raises(FileNotFoundError):
            parse_iemocap_session("/nonexistent/path")

    def test_load_multiple_sessions(self):
        """Test loading across multiple sessions."""
        from src.data.iemocap_loader import load_iemocap

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for s in [1, 2, 3]:
                self._create_mock_session(root / f"Session{s}", session_num=s)

            all_utts = load_iemocap(root, sessions=[1, 2, 3])
            assert len(all_utts) == 15  # 5 per session × 3

            # Test loading subset
            subset = load_iemocap(root, sessions=[1])
            assert len(subset) == 5

    def test_cv_splits(self):
        """Test cross-validation split generation."""
        from src.data.iemocap_loader import load_iemocap, get_iemocap_cv_splits

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for s in [1, 2, 3]:
                self._create_mock_session(root / f"Session{s}", session_num=s)

            utterances = load_iemocap(root, sessions=[1, 2, 3])
            splits = get_iemocap_cv_splits(utterances)

            # With sessions 1-3, we get 5 splits (leave-one-session-out for sessions 1-5)
            assert len(splits) == 5

            # Session 1 test fold should have 5 utterances
            assert len(splits[0]["test"]) == 5
            assert len(splits[0]["train"]) == 10


# ---------------------------------------------------------------------------
# RAVDESS loader tests
# ---------------------------------------------------------------------------

class TestRAVDESSLoader:
    """Tests for the RAVDESS dataset loader."""

    def _create_mock_ravdess(self, root_dir: Path) -> None:
        """Create a minimal mock RAVDESS directory."""
        # Actor_01 (male, odd)
        actor_dir = root_dir / "Actor_01"
        # 03 = audio-only, 01 = speech, XX = emotion code
        _write_wav(actor_dir / "03-01-01-01-01-01-01.wav")  # neutral
        _write_wav(actor_dir / "03-01-03-01-01-01-01.wav")  # happy
        _write_wav(actor_dir / "03-01-05-02-01-01-01.wav")  # angry, strong
        _write_wav(actor_dir / "03-01-06-01-01-01-01.wav")  # fear (should be filtered in 4-class)

        # Actor_02 (female, even)
        actor_dir2 = root_dir / "Actor_02"
        _write_wav(actor_dir2 / "03-01-04-01-01-01-02.wav")  # sad
        _write_wav(actor_dir2 / "03-01-02-01-01-01-02.wav")  # calm → neutral

        # Non-speech file (should be filtered)
        _write_wav(actor_dir / "03-02-01-01-01-01-01.wav")  # song

    def test_filename_parsing(self):
        """Test RAVDESS filename parsing."""
        from src.data.ravdess_loader import parse_ravdess_filename

        result = parse_ravdess_filename(Path("03-01-05-02-01-01-12.wav"))
        assert result is not None
        assert result["emotion"] == "angry"
        assert result["intensity"] == "strong"
        assert result["actor_id"] == 12
        assert result["gender"] == "female"  # even actor ID

    def test_load_4class(self):
        """Test loading with 4-class filter."""
        from src.data.ravdess_loader import load_ravdess

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._create_mock_ravdess(root)

            utterances = load_ravdess(root, four_class=True, speech_only=True)
            emotions = {u.emotion_4class for u in utterances}

            assert emotions <= {"angry", "happy", "sad", "neutral"}
            # fear should be excluded
            assert all(u.emotion != "fear" for u in utterances)

    def test_gender_assignment(self):
        """Test gender assignment based on actor ID parity."""
        from src.data.ravdess_loader import load_ravdess

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._create_mock_ravdess(root)

            utterances = load_ravdess(root, four_class=False, speech_only=True)

            for u in utterances:
                if u.actor_id % 2 == 1:
                    assert u.gender == "male"
                else:
                    assert u.gender == "female"

    def test_calm_merges_to_neutral(self):
        """Test that calm emotion maps to neutral in 4-class."""
        from src.data.ravdess_loader import load_ravdess

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._create_mock_ravdess(root)

            utterances = load_ravdess(root, four_class=True)
            # Actor_02 has a calm file → should appear as neutral
            actor2_utts = [u for u in utterances if u.actor_id == 2]
            assert any(u.emotion_4class == "neutral" for u in actor2_utts)

    def test_missing_dir_raises(self):
        """Test that missing directory raises FileNotFoundError."""
        from src.data.ravdess_loader import load_ravdess

        with pytest.raises(FileNotFoundError):
            load_ravdess("/nonexistent/path")


# ---------------------------------------------------------------------------
# CREMA-D loader tests
# ---------------------------------------------------------------------------

class TestCREMADLoader:
    """Tests for the CREMA-D dataset loader."""

    def _create_mock_cremad(self, root_dir: Path) -> None:
        """Create a minimal mock CREMA-D directory."""
        audio_dir = root_dir / "AudioWAV"
        _write_wav(audio_dir / "1001_DFA_ANG_XX.wav")
        _write_wav(audio_dir / "1001_IEO_HAP_HI.wav")
        _write_wav(audio_dir / "1002_DFA_SAD_LO.wav")
        _write_wav(audio_dir / "1002_IOM_NEU_MD.wav")
        _write_wav(audio_dir / "1003_TAI_DIS_XX.wav")  # disgust → filtered in 4-class
        _write_wav(audio_dir / "1003_TIE_FEA_XX.wav")  # fear → filtered in 4-class

    def test_filename_parsing(self):
        """Test CREMA-D filename parsing."""
        from src.data.cremad_loader import parse_cremad_filename

        result = parse_cremad_filename(Path("1001_DFA_ANG_XX.wav"))
        assert result is not None
        assert result["actor_id"] == 1001
        assert result["sentence_code"] == "DFA"
        assert result["emotion"] == "angry"
        assert result["intensity"] == "unspecified"

    def test_load_4class(self):
        """Test loading with 4-class filter."""
        from src.data.cremad_loader import load_cremad

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._create_mock_cremad(root)

            utterances = load_cremad(root, four_class=True)
            assert len(utterances) == 4  # ANG, HAP, SAD, NEU
            emotions = {u.emotion_4class for u in utterances}
            assert emotions <= {"angry", "happy", "sad", "neutral"}

    def test_load_all_classes(self):
        """Test loading without 4-class filter includes all emotions."""
        from src.data.cremad_loader import load_cremad

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._create_mock_cremad(root)

            utterances = load_cremad(root, four_class=False)
            assert len(utterances) == 6  # All 6 files

    def test_missing_dir_raises(self):
        from src.data.cremad_loader import load_cremad

        with pytest.raises(FileNotFoundError):
            load_cremad("/nonexistent/path")


# ---------------------------------------------------------------------------
# MSP-IMPROV loader tests
# ---------------------------------------------------------------------------

class TestMSPImprovLoader:
    """Tests for the MSP-IMPROV dataset loader."""

    def _create_mock_msp(self, root_dir: Path) -> None:
        """Create a minimal mock MSP-IMPROV directory."""
        audio_dir = root_dir / "Audios"
        audio_dir.mkdir(parents=True, exist_ok=True)

        # Create audio files
        _write_wav(audio_dir / "utt001.wav")
        _write_wav(audio_dir / "utt002.wav")
        _write_wav(audio_dir / "utt003.wav")
        _write_wav(audio_dir / "utt004.wav")

        # Create labels file
        labels_file = root_dir / "labels.csv"
        labels_file.write_text(
            "utterance_id,emotion,speaker_id,gender,session_id\n"
            "utt001,angry,spk01,male,session1\n"
            "utt002,happy,spk01,male,session1\n"
            "utt003,sad,spk02,female,session2\n"
            "utt004,neutral,spk02,female,session2\n"
        )

    def test_load_basic(self):
        """Test basic MSP-IMPROV loading."""
        from src.data.msp_improv_loader import load_msp_improv

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._create_mock_msp(root)

            utterances = load_msp_improv(root)
            assert len(utterances) == 4

            emotions = {u.emotion for u in utterances}
            assert emotions == {"angry", "happy", "sad", "neutral"}

    def test_missing_labels_raises(self):
        from src.data.msp_improv_loader import load_msp_improv

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "Audios").mkdir(parents=True)
            _write_wav(root / "Audios" / "utt001.wav")

            with pytest.raises(FileNotFoundError, match="labels"):
                load_msp_improv(root)

    def test_missing_dir_raises(self):
        from src.data.msp_improv_loader import load_msp_improv

        with pytest.raises(FileNotFoundError):
            load_msp_improv("/nonexistent/path")


# ---------------------------------------------------------------------------
# GoEmotions loader tests
# ---------------------------------------------------------------------------

class TestGoEmotionsLoader:
    """Tests for the GoEmotions dataset loader."""

    def test_4class_mapping(self):
        """Test that GoEmotions fine labels map correctly to 4-class."""
        from src.data.goemotions_loader import _resolve_4class

        assert _resolve_4class(["joy"]) == "happy"
        assert _resolve_4class(["anger"]) == "angry"
        assert _resolve_4class(["sadness"]) == "sad"
        assert _resolve_4class(["neutral"]) == "neutral"
        assert _resolve_4class(["excitement", "joy"]) == "happy"
        assert _resolve_4class(["fear"]) is None  # Unmappable

    def test_majority_vote(self):
        """Test majority vote with mixed labels."""
        from src.data.goemotions_loader import _resolve_4class

        # Two happy labels vs one angry → happy wins
        result = _resolve_4class(["joy", "excitement", "anger"])
        assert result == "happy"

    def test_load_from_local_tsv(self):
        """Test loading GoEmotions from local TSV file."""
        from src.data.goemotions_loader import load_goemotions_from_local

        with tempfile.TemporaryDirectory() as tmpdir:
            tsv_path = Path(tmpdir) / "train.tsv"
            tsv_path.write_text(
                "text\tlabels\tid\n"
                "This is amazing!\t0,17\t1\n"  # admiration(0), joy(17) → happy
                "I hate this.\t2,3\t2\n"       # anger(2), annoyance(3) → angry
                "So sad today.\t25\t3\n"        # sadness(25) → sad
                "Nothing happened.\t27\t4\n"    # neutral(27) → neutral
                "I'm scared.\t14\t5\n"          # fear(14) → None (filtered)
            )

            records = load_goemotions_from_local(tmpdir, split="train", four_class=True)
            assert len(records) == 4  # fear is filtered out
            emotions = {r.emotion_4class for r in records}
            assert emotions == {"happy", "angry", "sad", "neutral"}

    def test_missing_file_raises(self):
        from src.data.goemotions_loader import load_goemotions_from_local

        with pytest.raises(FileNotFoundError):
            load_goemotions_from_local("/nonexistent/dir", split="train")


# ---------------------------------------------------------------------------
# Dataset Registry tests
# ---------------------------------------------------------------------------

class TestDatasetRegistry:
    """Tests for the unified dataset registry."""

    def test_supported_datasets(self):
        from src.data.dataset_registry import SUPPORTED_DATASETS

        assert "iemocap" in SUPPORTED_DATASETS
        assert "ravdess" in SUPPORTED_DATASETS
        assert "cremad" in SUPPORTED_DATASETS
        assert "msp_improv" in SUPPORTED_DATASETS
        assert "goemotions" in SUPPORTED_DATASETS

    def test_unknown_dataset_raises(self):
        from src.data.dataset_registry import load_dataset_by_name

        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset_by_name("nonexistent", "/tmp")

    def test_emotion_constants(self):
        from src.data.dataset_registry import EMOTION_LABELS_4CLASS, EMOTION_TO_IDX, IDX_TO_EMOTION

        assert len(EMOTION_LABELS_4CLASS) == 4
        assert EMOTION_TO_IDX["angry"] == 0
        assert EMOTION_TO_IDX["happy"] == 1
        assert EMOTION_TO_IDX["sad"] == 2
        assert EMOTION_TO_IDX["neutral"] == 3
        assert IDX_TO_EMOTION[0] == "angry"

    def test_class_distribution(self):
        from src.data.dataset_registry import EmotionRecord, get_class_distribution

        records = [
            EmotionRecord("a", "test", None, None, "happy", "happy", "s1", "M", "1"),
            EmotionRecord("b", "test", None, None, "happy", "happy", "s1", "M", "1"),
            EmotionRecord("c", "test", None, None, "angry", "angry", "s2", "F", "1"),
        ]
        dist = get_class_distribution(records)
        assert dist["happy"] == 2
        assert dist["angry"] == 1

    def test_records_to_dataframe(self):
        from src.data.dataset_registry import EmotionRecord, records_to_dataframe

        records = [
            EmotionRecord("a", "test", "/path.wav", None, "happy", "happy", "s1", "M", "1"),
        ]
        df = records_to_dataframe(records)
        assert len(df) == 1
        assert "file_id" in df.columns
        assert df.iloc[0]["emotion_4class"] == "happy"


# ---------------------------------------------------------------------------
# Audio utils tests
# ---------------------------------------------------------------------------

class TestAudioUtils:
    """Tests for audio preprocessing utilities."""

    def test_load_and_preprocess(self):
        from src.data.audio_utils import load_and_preprocess

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "test.wav"
            _write_wav(wav_path, sr=44100, duration_sec=2.0)  # Different SR

            audio = load_and_preprocess(wav_path)
            assert audio.dtype == np.float32
            assert len(audio) <= 16000 * 15  # Max 15 seconds at 16kHz
            assert np.max(np.abs(audio)) <= 1.0 + 1e-6  # Normalized

    def test_truncation(self):
        from src.data.audio_utils import load_and_preprocess

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "long.wav"
            _write_wav(wav_path, sr=16000, duration_sec=20.0)

            audio = load_and_preprocess(wav_path, max_duration_sec=5.0)
            assert len(audio) <= 16000 * 5

    def test_pad_or_truncate(self):
        from src.data.audio_utils import pad_or_truncate

        short = np.ones(100, dtype=np.float32)
        padded = pad_or_truncate(short, 200)
        assert len(padded) == 200
        assert padded[100] == 0.0

        long = np.ones(300, dtype=np.float32)
        truncated = pad_or_truncate(long, 200)
        assert len(truncated) == 200

    def test_missing_file_raises(self):
        from src.data.audio_utils import load_and_preprocess

        with pytest.raises(FileNotFoundError):
            load_and_preprocess("/nonexistent/audio.wav")

    def test_validate_audio(self):
        from src.data.audio_utils import validate_audio

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "test.wav"
            _write_wav(wav_path, duration_sec=1.0)

            assert validate_audio(wav_path) is True
            assert validate_audio("/nonexistent.wav") is False


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    """Tests for configuration loading."""

    def test_load_default_config(self):
        from src.utils.config import load_config

        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
        if not config_path.exists():
            pytest.skip("default.yaml not found")

        config = load_config(config_path)
        assert config.seed == 42
        assert config.data.sample_rate == 16000
        assert config.pipeline_a.model_name == "microsoft/deberta-v3-base"
        assert config.pipeline_b.tcn_blocks == 6
        assert config.fusion.embed_dim == 256

    def test_config_overrides(self):
        from src.utils.config import load_config

        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
        if not config_path.exists():
            pytest.skip("default.yaml not found")

        config = load_config(config_path, overrides={"seed": 123, "pipeline_a.learning_rate": 1e-4})
        assert config.seed == 123
        assert config.pipeline_a.learning_rate == 1e-4

    def test_save_and_reload(self):
        from src.utils.config import MERAConfig, save_config, load_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MERAConfig(seed=99)
            save_path = Path(tmpdir) / "test_config.yaml"
            save_config(config, save_path)

            reloaded = load_config(save_path)
            assert reloaded.seed == 99
