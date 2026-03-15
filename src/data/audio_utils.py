"""Audio loading, resampling, and normalization utilities.

All audio in the MERA pipeline is standardized to:
- Sample rate: 16000 Hz
- Channels: mono
- Bit depth: float32 (normalized to [-1.0, 1.0])
- Silence trimming: leading/trailing silence removed (top_db=20)
- Max duration: 15 seconds (truncate longer, pad shorter for batching)
"""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

TARGET_SR = 16000
MAX_DURATION_SEC = 15.0
MAX_SAMPLES = int(TARGET_SR * MAX_DURATION_SEC)


def load_and_preprocess(
    audio_path: str | Path,
    sr: int = TARGET_SR,
    max_duration_sec: float = MAX_DURATION_SEC,
    trim_silence: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Load an audio file, resample, normalize, and trim silence.

    Args:
        audio_path: Path to the audio file (WAV, FLAC, etc.).
        sr: Target sample rate. Defaults to 16000.
        max_duration_sec: Maximum duration in seconds; longer clips
            are truncated. Defaults to 15.0.
        trim_silence: Whether to trim leading/trailing silence.
        normalize: Whether to normalize amplitude to [-1, 1].

    Returns:
        1-D float32 numpy array of the preprocessed audio waveform.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        RuntimeError: If the audio file cannot be loaded.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        y, orig_sr = librosa.load(str(audio_path), sr=sr, mono=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio {audio_path}: {e}") from e

    # Trim leading/trailing silence
    if trim_silence and len(y) > 0:
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        if len(y_trimmed) > 0:
            y = y_trimmed

    # Normalize to [-1, 1]
    if normalize:
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak

    # Truncate if too long
    max_samples = int(sr * max_duration_sec)
    if len(y) > max_samples:
        y = y[:max_samples]

    return y.astype(np.float32)


def pad_or_truncate(
    audio: np.ndarray,
    target_length: int,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Pad or truncate an audio array to a fixed length.

    Args:
        audio: 1-D audio waveform.
        target_length: Desired number of samples.
        pad_value: Value used for padding. Defaults to 0.0.

    Returns:
        Audio array of exactly ``target_length`` samples.
    """
    if len(audio) >= target_length:
        return audio[:target_length]
    padding = np.full(target_length - len(audio), pad_value, dtype=audio.dtype)
    return np.concatenate([audio, padding])


def get_audio_info(audio_path: str | Path) -> dict:
    """Get metadata about an audio file without loading the full waveform.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Dict with keys: sample_rate, channels, duration_sec, frames.
    """
    audio_path = Path(audio_path)
    info = sf.info(str(audio_path))
    return {
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "duration_sec": info.duration,
        "frames": info.frames,
    }


def validate_audio(audio_path: str | Path, min_duration_sec: float = 0.1) -> bool:
    """Check whether an audio file is valid and long enough.

    Args:
        audio_path: Path to the audio file.
        min_duration_sec: Minimum acceptable duration. Defaults to 0.1s.

    Returns:
        True if the file is a valid audio file exceeding the minimum duration.
    """
    try:
        info = get_audio_info(audio_path)
        return info["duration_sec"] >= min_duration_sec
    except Exception:
        return False
