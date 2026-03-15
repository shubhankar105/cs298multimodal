"""Log-Mel spectrogram extraction for the CNN-BiLSTM stream.

Parameters chosen to match speech processing conventions:
- 128 mel bands (high resolution for emotion-relevant formants)
- 25 ms window (400 samples at 16 kHz)
- 10 ms hop  (160 samples at 16 kHz)
- Frequency range: 20 Hz – 8 000 Hz (covers full speech range)
- FFT window: 512 (zero-padded from 25 ms for sharper frequency bins)

Output shape: ``(n_mels, n_frames) = (128, T)`` where *T* varies with
duration.  For 5 s audio *T* ≈ 313; for 10 s audio *T* ≈ 626.

Stored as **float32** NumPy arrays.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np

from src.data.audio_utils import load_and_preprocess

logger = logging.getLogger(__name__)

# Default extraction parameters (must agree with configs/default.yaml)
N_MELS = 128
N_FFT = 512
HOP_LENGTH = 160      # 10 ms hop
WIN_LENGTH = 400      # 25 ms window
F_MIN = 20
F_MAX = 8000
SR = 16000


def extract_log_mel(
    audio: np.ndarray,
    sr: int = SR,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    win_length: int = WIN_LENGTH,
    f_min: int = F_MIN,
    f_max: int = F_MAX,
) -> np.ndarray:
    """Extract a log-Mel spectrogram from a preprocessed audio waveform.

    Args:
        audio: 1-D float32 array of the waveform (already at *sr* Hz).
        sr: Sample rate.  Defaults to 16 000.
        n_mels: Number of Mel filter-bank bands.
        n_fft: FFT window size.
        hop_length: Hop length in samples.
        win_length: Analysis window length in samples.
        f_min: Lowest frequency for the Mel filter-bank.
        f_max: Highest frequency for the Mel filter-bank.

    Returns:
        ``np.ndarray`` of shape ``(n_mels, n_frames)``, dtype float32,
        normalised to the ``[0, 1]`` range.
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
        power=2.0,
    )

    # Log transform with small epsilon for numerical stability
    log_mel = librosa.power_to_db(mel_spec, ref=np.max, top_db=80.0)

    # Normalise to [0, 1]
    mel_min = log_mel.min()
    mel_max = log_mel.max()
    log_mel = (log_mel - mel_min) / (mel_max - mel_min + 1e-8)

    return log_mel.astype(np.float32)


def extract_and_save(
    audio_path: str | Path,
    output_path: str | Path,
    sr: int = SR,
    max_duration_sec: float = 15.0,
) -> np.ndarray:
    """Load audio, extract log-Mel, and save to ``.npy``.

    Args:
        audio_path: Path to the source audio file.
        output_path: Destination ``.npy`` path.
        sr: Target sample rate.
        max_duration_sec: Maximum audio duration in seconds.

    Returns:
        The extracted log-Mel spectrogram array.
    """
    audio = load_and_preprocess(audio_path, sr=sr, max_duration_sec=max_duration_sec)
    log_mel = extract_log_mel(audio, sr=sr)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), log_mel)
    return log_mel


def extract_spectrograms_batch(
    audio_paths: List[str | Path],
    output_dir: str | Path,
    sr: int = SR,
    max_duration_sec: float = 15.0,
    show_progress: bool = True,
) -> dict:
    """Batch-extract log-Mel spectrograms, with skip-if-exists logic.

    Args:
        audio_paths: List of audio file paths.
        output_dir: Directory to store ``.npy`` files.
        sr: Target sample rate.
        max_duration_sec: Maximum audio duration.
        show_progress: Whether to display a tqdm progress bar.

    Returns:
        Dict with ``processed``, ``skipped``, ``errors`` counts.
    """
    from tqdm import tqdm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"processed": 0, "skipped": 0, "errors": []}
    iterator = tqdm(audio_paths, desc="Spectrograms", disable=not show_progress)

    for audio_path in iterator:
        file_id = Path(audio_path).stem
        output_path = output_dir / f"{file_id}.npy"

        if output_path.exists():
            stats["skipped"] += 1
            continue

        try:
            extract_and_save(audio_path, output_path, sr=sr, max_duration_sec=max_duration_sec)
            stats["processed"] += 1
        except Exception as e:
            logger.error(f"Failed spectrogram for {file_id}: {e}")
            stats["errors"].append(str(audio_path))

    logger.info(
        f"Spectrogram extraction: {stats['processed']} processed, "
        f"{stats['skipped']} skipped, {len(stats['errors'])} errors"
    )
    return stats
