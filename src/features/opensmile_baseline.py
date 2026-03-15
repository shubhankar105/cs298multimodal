"""eGeMAPS baseline feature extraction via openSMILE.

Extracts the extended Geneva Minimalistic Acoustic Parameter Set (eGeMAPS)
for baseline comparisons in ablation studies.  eGeMAPS produces an
88-dimensional feature vector summarising an entire utterance, which is
the standard approach that our prosodic-contour TCN aims to outperform.

Supports two modes:
1. **Utterance-level** (default): one 88-d vector per file (for SVM / MLP baselines).
2. **Frame-level**: per-frame Low-Level Descriptors (LLDs) for richer baselines.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def extract_egemaps(
    audio_path: str | Path,
    level: str = "functionals",
) -> np.ndarray:
    """Extract eGeMAPS features from a single audio file.

    Args:
        audio_path: Path to the audio file.
        level: ``"functionals"`` for utterance-level (88-d) or
               ``"lld"`` for frame-level Low-Level Descriptors.

    Returns:
        For functionals: ``np.ndarray`` of shape ``(88,)`` (float32).
        For LLD: ``np.ndarray`` of shape ``(T, D)`` where *D* depends
        on the LLD feature set.

    Raises:
        ImportError: If opensmile is not installed.
    """
    try:
        import opensmile
    except ImportError:
        raise ImportError(
            "opensmile is required for eGeMAPS extraction. "
            "Install with: pip install opensmile"
        )

    feature_level = (
        opensmile.FeatureLevel.Functionals
        if level == "functionals"
        else opensmile.FeatureLevel.LowLevelDescriptors
    )

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=feature_level,
    )

    df = smile.process_file(str(audio_path))
    features = df.values.astype(np.float32)

    if level == "functionals":
        # Flatten to 1-D: (1, 88) → (88,)
        return features.squeeze(0)

    # LLD: shape (T, D)
    return features


def extract_egemaps_batch(
    audio_paths: List[str | Path],
    output_dir: str | Path,
    level: str = "functionals",
    show_progress: bool = True,
) -> dict:
    """Batch-extract eGeMAPS features with skip-if-exists.

    Args:
        audio_paths: List of audio file paths.
        output_dir: Directory for ``.npy`` output files.
        level: ``"functionals"`` or ``"lld"``.
        show_progress: Show tqdm progress bar.

    Returns:
        Dict with ``processed``, ``skipped``, ``errors`` counts.
    """
    from tqdm import tqdm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"processed": 0, "skipped": 0, "errors": []}
    iterator = tqdm(audio_paths, desc="eGeMAPS", disable=not show_progress)

    for audio_path in iterator:
        file_id = Path(audio_path).stem
        output_path = output_dir / f"{file_id}.npy"

        if output_path.exists():
            stats["skipped"] += 1
            continue

        try:
            features = extract_egemaps(audio_path, level=level)
            np.save(str(output_path), features)
            stats["processed"] += 1
        except Exception as e:
            logger.error(f"Failed eGeMAPS for {file_id}: {e}")
            stats["errors"].append(str(audio_path))

    logger.info(
        f"eGeMAPS extraction: {stats['processed']} processed, "
        f"{stats['skipped']} skipped, {len(stats['errors'])} errors"
    )
    return stats


def load_egemaps_matrix(
    npy_paths: List[str | Path],
) -> tuple[np.ndarray, List[str]]:
    """Load multiple eGeMAPS ``.npy`` files into a feature matrix.

    Useful for training SVM / MLP baselines.

    Args:
        npy_paths: List of paths to ``.npy`` files (functionals-level).

    Returns:
        Tuple of:
        - Feature matrix ``(N, 88)`` float32.
        - List of file IDs (stems).
    """
    features = []
    file_ids = []
    for path in npy_paths:
        path = Path(path)
        feat = np.load(str(path)).astype(np.float32)
        if feat.ndim == 1:
            features.append(feat)
        else:
            features.append(feat.squeeze(0))
        file_ids.append(path.stem)

    return np.stack(features, axis=0), file_ids
