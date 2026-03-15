"""NOVEL CONTRIBUTION: Frame-level prosodic contour extraction.

Instead of extracting summary statistics (mean, std, range) from prosodic
features—the standard approach in **all** existing SER literature—we treat
prosodic contours as continuous 1-D signals.  The downstream Temporal
Convolutional Network (TCN) then learns emotion-discriminative temporal
patterns directly from the *shape* of these trajectories.

Features extracted per frame (10 ms resolution, aligned with spectrograms):

.. list-table::
   :header-rows: 1

   * - Channel
     - Feature
     - Method
   * - 0
     - F0 (fundamental frequency / pitch)
     - Praat autocorrelation
   * - 1
     - Log Energy (RMS in dB)
     - librosa RMS → amplitude_to_db
   * - 2
     - Spectral Centroid
     - librosa
   * - 3
     - HNR (Harmonics-to-Noise Ratio)
     - Praat cross-correlation
   * - 4
     - Jitter (local, windowed)
     - Praat PointProcess
   * - 5
     - Shimmer (local, windowed)
     - Praat PointProcess
   * - 6
     - Formant F1
     - Praat Burg
   * - 7
     - Formant F2
     - Praat Burg
   * - 8
     - Formant F3
     - Praat Burg
   * - 9
     - MFCC Δ (1st derivative of MFCC₀)
     - librosa delta

Total: **10 contour channels** per frame.
Output shape: ``(10, T)`` where ``T = n_frames`` (same as the spectrogram).

All contours are **per-utterance zero-mean, unit-variance normalised** after
extraction.  Unvoiced frames retain ``F0 = 0`` (the TCN learns to handle this).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call

from src.data.audio_utils import load_and_preprocess

logger = logging.getLogger(__name__)

# Extraction parameters — must match spectrogram hop/win
SR = 16000
HOP_LENGTH = 160          # 10 ms hop → aligns with spectrogram frames
WIN_LENGTH = 400          # 25 ms window
FRAME_DURATION = HOP_LENGTH / SR   # 0.01 s

# Praat pitch tracking bounds (covers normal + emotional speech)
PITCH_FLOOR = 75.0        # Hz
PITCH_CEILING = 600.0     # Hz

# Jitter/shimmer analysis window (100 ms centred on each frame)
PERTURBATION_WINDOW = 0.1  # seconds

# Number of output channels
NUM_CHANNELS = 10

# Feature channel names (for logging / debugging)
CHANNEL_NAMES = [
    "F0", "LogEnergy", "SpectralCentroid", "HNR",
    "Jitter", "Shimmer", "F1", "F2", "F3", "MFCCDelta",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_prosodic_contours(
    audio: np.ndarray,
    sr: int = SR,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Extract 10-channel frame-level prosodic contours from audio.

    Args:
        audio: 1-D float32 preprocessed waveform.
        sr: Sample rate (must match *audio*).
        hop_length: Hop length in samples (controls frame rate).

    Returns:
        ``np.ndarray`` of shape ``(10, n_frames)``, dtype float32,
        per-channel zero-mean / unit-variance normalised.
    """
    duration = len(audio) / sr
    n_frames = 1 + (len(audio) - 1) // hop_length
    frame_times = np.arange(n_frames) * (hop_length / sr)

    # Create Praat Sound object (required for pitch / formant / HNR)
    snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sr)

    # --- Channel 0: F0 (pitch) ---
    f0 = _extract_f0(snd, frame_times)

    # --- Channel 1: Log energy (RMS in dB) ---
    log_energy = _extract_log_energy(audio, n_frames, hop_length=hop_length)

    # --- Channel 2: Spectral centroid ---
    spectral_centroid = _extract_spectral_centroid(audio, sr, n_frames, hop_length)

    # --- Channel 3: HNR ---
    hnr = _extract_hnr(snd, frame_times)

    # --- Channels 4–5: Jitter & shimmer (windowed) ---
    jitter, shimmer = _extract_perturbations(snd, frame_times, duration)

    # --- Channels 6–8: Formants F1, F2, F3 ---
    f1, f2, f3 = _extract_formants(snd, frame_times)

    # --- Channel 9: MFCC delta ---
    mfcc_delta = _extract_mfcc_delta(audio, sr, n_frames, hop_length)

    # Stack: (10, n_frames)
    contours = np.stack(
        [f0, log_energy, spectral_centroid, hnr, jitter, shimmer, f1, f2, f3, mfcc_delta],
        axis=0,
    )

    # Per-channel z-normalisation
    contours = _normalise_channels(contours)

    return contours.astype(np.float32)


def extract_and_save(
    audio_path: str | Path,
    output_path: str | Path,
    sr: int = SR,
    max_duration_sec: float = 15.0,
) -> np.ndarray:
    """Load audio, extract prosodic contours, save to ``.npy``.

    Args:
        audio_path: Path to source audio.
        output_path: Destination ``.npy`` file.
        sr: Target sample rate.
        max_duration_sec: Maximum audio duration.

    Returns:
        The extracted contours array ``(10, T)``.
    """
    audio = load_and_preprocess(audio_path, sr=sr, max_duration_sec=max_duration_sec)
    contours = extract_prosodic_contours(audio, sr=sr)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), contours)
    return contours


def extract_prosodics_batch(
    audio_paths: List[str | Path],
    output_dir: str | Path,
    sr: int = SR,
    max_duration_sec: float = 15.0,
    show_progress: bool = True,
) -> dict:
    """Batch-extract prosodic contours with skip-if-exists.

    Args:
        audio_paths: Audio file paths.
        output_dir: Directory for ``.npy`` outputs.
        sr: Target sample rate.
        max_duration_sec: Max audio duration.
        show_progress: Show tqdm bar.

    Returns:
        Dict with ``processed``, ``skipped``, ``errors`` counts.
    """
    from tqdm import tqdm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"processed": 0, "skipped": 0, "errors": []}
    iterator = tqdm(audio_paths, desc="Prosodic contours", disable=not show_progress)

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
            logger.error(f"Failed prosodic extraction for {file_id}: {e}")
            stats["errors"].append(str(audio_path))

    logger.info(
        f"Prosodic extraction: {stats['processed']} processed, "
        f"{stats['skipped']} skipped, {len(stats['errors'])} errors"
    )
    return stats


# ---------------------------------------------------------------------------
# Private extraction helpers
# ---------------------------------------------------------------------------

def _extract_f0(snd: parselmouth.Sound, frame_times: np.ndarray) -> np.ndarray:
    """Extract F0 contour via Praat autocorrelation method."""
    pitch = call(snd, "To Pitch", 0.0, PITCH_FLOOR, PITCH_CEILING)
    f0 = np.array(
        [call(pitch, "Get value at time", float(t), "Hertz", "linear") for t in frame_times]
    )
    return np.nan_to_num(f0, nan=0.0)


def _extract_log_energy(
    audio: np.ndarray,
    n_frames: int,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Extract frame-level log-RMS energy in dB."""
    rms = librosa.feature.rms(y=audio, frame_length=WIN_LENGTH, hop_length=hop_length)[0]
    rms = _align_to_frames(rms, n_frames)
    return librosa.amplitude_to_db(rms + 1e-8)


def _extract_spectral_centroid(
    audio: np.ndarray,
    sr: int,
    n_frames: int,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Extract spectral centroid (brightness measure)."""
    sc = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
    return _align_to_frames(sc, n_frames)


def _extract_hnr(snd: parselmouth.Sound, frame_times: np.ndarray) -> np.ndarray:
    """Extract Harmonics-to-Noise Ratio via Praat."""
    harmonicity = call(
        snd, "To Harmonicity (cc)", FRAME_DURATION, PITCH_FLOOR, 0.1, 1.0
    )
    hnr = np.array(
        [call(harmonicity, "Get value at time", float(t), "cubic") for t in frame_times]
    )
    return np.nan_to_num(hnr, nan=0.0)


def _extract_perturbations(
    snd: parselmouth.Sound,
    frame_times: np.ndarray,
    duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract windowed jitter and shimmer contours."""
    point_process = call(snd, "To PointProcess (periodic, cc)", PITCH_FLOOR, PITCH_CEILING)

    jitter = _compute_windowed_jitter(point_process, frame_times, duration)
    shimmer = _compute_windowed_shimmer(snd, point_process, frame_times, duration)
    return jitter, shimmer


def _compute_windowed_jitter(
    point_process,
    frame_times: np.ndarray,
    duration: float,
    window: float = PERTURBATION_WINDOW,
) -> np.ndarray:
    """Compute local jitter in overlapping windows centred at each frame time."""
    jitter = np.zeros(len(frame_times), dtype=np.float64)
    half = window / 2
    for i, t in enumerate(frame_times):
        t_start = max(0.0, t - half)
        t_end = min(duration, t + half)
        try:
            j = call(
                point_process,
                "Get jitter (local)",
                t_start, t_end,
                0.0001, 0.02, 1.3,
            )
            jitter[i] = j if not np.isnan(j) else 0.0
        except Exception:
            jitter[i] = 0.0
    return jitter


def _compute_windowed_shimmer(
    snd: parselmouth.Sound,
    point_process,
    frame_times: np.ndarray,
    duration: float,
    window: float = PERTURBATION_WINDOW,
) -> np.ndarray:
    """Compute local shimmer in overlapping windows centred at each frame time."""
    shimmer = np.zeros(len(frame_times), dtype=np.float64)
    half = window / 2
    for i, t in enumerate(frame_times):
        t_start = max(0.0, t - half)
        t_end = min(duration, t + half)
        try:
            s = call(
                [snd, point_process],
                "Get shimmer (local)",
                t_start, t_end,
                0.0001, 0.02, 1.3, 1.6,
            )
            shimmer[i] = s if not np.isnan(s) else 0.0
        except Exception:
            shimmer[i] = 0.0
    return shimmer


def _extract_formants(
    snd: parselmouth.Sound,
    frame_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract F1, F2, F3 formant contours via Praat Burg method."""
    formants = call(
        snd, "To Formant (burg)",
        FRAME_DURATION,   # time step
        5,                # max number of formants
        5500.0,           # max formant frequency
        0.025,            # window length
        50.0,             # pre-emphasis from
    )
    f1 = np.array(
        [call(formants, "Get value at time", 1, float(t), "hertz", "linear") for t in frame_times]
    )
    f2 = np.array(
        [call(formants, "Get value at time", 2, float(t), "hertz", "linear") for t in frame_times]
    )
    f3 = np.array(
        [call(formants, "Get value at time", 3, float(t), "hertz", "linear") for t in frame_times]
    )
    return np.nan_to_num(f1, nan=0.0), np.nan_to_num(f2, nan=0.0), np.nan_to_num(f3, nan=0.0)


def _extract_mfcc_delta(
    audio: np.ndarray,
    sr: int,
    n_frames: int,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Extract 1st-order delta of MFCC₀ (energy dynamics)."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=1, hop_length=hop_length)
    mfcc_delta = librosa.feature.delta(mfcc, order=1)[0]
    return _align_to_frames(mfcc_delta, n_frames)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _align_to_frames(feature: np.ndarray, n_frames: int) -> np.ndarray:
    """Pad or truncate a 1-D feature vector to exactly *n_frames*."""
    if len(feature) >= n_frames:
        return feature[:n_frames]
    return np.pad(feature, (0, n_frames - len(feature)), mode="edge")


def _normalise_channels(contours: np.ndarray) -> np.ndarray:
    """Per-channel zero-mean unit-variance normalisation (in-place)."""
    for i in range(contours.shape[0]):
        mean = contours[i].mean()
        std = contours[i].std()
        if std > 1e-8:
            contours[i] = (contours[i] - mean) / std
        else:
            contours[i] = contours[i] - mean
    return contours
