#!/usr/bin/env python3
"""Master preprocessing script for MERA.

Runs all feature extraction stages in sequence:
1. Whisper transcription  → ``data/processed/transcripts/{file_id}.json``
2. Spectrogram extraction → ``data/processed/spectrograms/{file_id}.npy``
3. Prosodic contour extraction → ``data/processed/prosodic_contours/{file_id}.npy``
4. HuBERT embedding extraction → ``data/processed/hubert_embeddings/{file_id}.npy``

Generates a master metadata CSV at ``data/processed/metadata.csv`` with
columns: file_id, dataset, audio_path, emotion_4class, speaker_id,
spectrogram_path, prosody_path, hubert_path, transcript_path, duration_sec.

All stages support **skip-if-exists** for resumable runs.

Usage::

    python scripts/preprocess_all.py --config configs/default.yaml \\
                                      --datasets iemocap,ravdess,cremad

    # Resume a partially completed run:
    python scripts/preprocess_all.py --config configs/default.yaml \\
                                      --skip-whisper --skip-spectrogram
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.device import configure_mps_memory, get_device
from src.utils.seed import seed_everything
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Default directories (relative to project root)
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_PROCESSED = DEFAULT_DATA_ROOT / "processed"
TRANSCRIPT_DIR = DEFAULT_PROCESSED / "transcripts"
SPECTROGRAM_DIR = DEFAULT_PROCESSED / "spectrograms"
PROSODIC_DIR = DEFAULT_PROCESSED / "prosodic_contours"
HUBERT_DIR = DEFAULT_PROCESSED / "hubert_embeddings"
METADATA_PATH = DEFAULT_PROCESSED / "metadata.csv"


def collect_audio_records(
    datasets: list[str],
    data_raw_dir: Path,
) -> list[dict]:
    """Gather audio file records from all requested datasets.

    Returns a list of dicts with keys: file_id, dataset, audio_path,
    emotion_4class, speaker_id.
    """
    records: list[dict] = []

    if "iemocap" in datasets:
        from src.data.iemocap_loader import load_iemocap

        iemocap_dir = data_raw_dir / "iemocap"
        if iemocap_dir.exists():
            utts = load_iemocap(iemocap_dir)
            for u in utts:
                records.append({
                    "file_id": u.utterance_id,
                    "dataset": "iemocap",
                    "audio_path": u.audio_path,
                    "emotion_4class": u.emotion,
                    "speaker_id": u.speaker_id,
                })
            logger.info(f"Collected {len(utts)} IEMOCAP utterances")
        else:
            logger.warning(f"IEMOCAP directory not found: {iemocap_dir}")

    if "ravdess" in datasets:
        from src.data.ravdess_loader import load_ravdess

        ravdess_dir = data_raw_dir / "ravdess"
        if ravdess_dir.exists():
            utts = load_ravdess(ravdess_dir, four_class=True)
            for u in utts:
                records.append({
                    "file_id": u.utterance_id,
                    "dataset": "ravdess",
                    "audio_path": u.audio_path,
                    "emotion_4class": u.emotion_4class,
                    "speaker_id": u.speaker_id,
                })
            logger.info(f"Collected {len(utts)} RAVDESS utterances")
        else:
            logger.warning(f"RAVDESS directory not found: {ravdess_dir}")

    if "cremad" in datasets:
        from src.data.cremad_loader import load_cremad

        cremad_dir = data_raw_dir / "cremad"
        if cremad_dir.exists():
            utts = load_cremad(cremad_dir, four_class=True)
            for u in utts:
                records.append({
                    "file_id": u.utterance_id,
                    "dataset": "cremad",
                    "audio_path": u.audio_path,
                    "emotion_4class": u.emotion_4class,
                    "speaker_id": u.speaker_id,
                })
            logger.info(f"Collected {len(utts)} CREMA-D utterances")
        else:
            logger.warning(f"CREMA-D directory not found: {cremad_dir}")

    if "msp_improv" in datasets:
        from src.data.msp_improv_loader import load_msp_improv

        msp_dir = data_raw_dir / "msp_improv"
        if msp_dir.exists():
            try:
                utts = load_msp_improv(msp_dir, four_class=True)
                for u in utts:
                    records.append({
                        "file_id": u.utterance_id,
                        "dataset": "msp_improv",
                        "audio_path": u.audio_path,
                        "emotion_4class": u.emotion_4class,
                        "speaker_id": u.speaker_id,
                    })
                logger.info(f"Collected {len(utts)} MSP-IMPROV utterances")
            except FileNotFoundError as e:
                logger.warning(f"MSP-IMPROV labels not found: {e}")
        else:
            logger.warning(f"MSP-IMPROV directory not found: {msp_dir}")

    logger.info(f"Total audio records collected: {len(records)}")
    return records


def get_audio_duration(audio_path: str) -> float:
    """Get duration in seconds (fast, no full load)."""
    try:
        from src.data.audio_utils import get_audio_info

        info = get_audio_info(audio_path)
        return info["duration_sec"]
    except Exception:
        return 0.0


# ---- Stage runners ----

def run_whisper_stage(audio_paths: list[str], output_dir: Path, backend: str = "mlx") -> dict:
    """Stage 1: Whisper transcription."""
    from src.features.whisper_transcriber import transcribe_batch

    logger.info("=" * 60)
    logger.info(f"STAGE 1: Whisper Transcription (backend={backend})")
    logger.info("=" * 60)
    start = time.time()
    stats = transcribe_batch(audio_paths, output_dir, show_progress=True, backend=backend)
    logger.info(f"Whisper stage completed in {time.time() - start:.1f}s")
    return stats


def run_spectrogram_stage(audio_paths: list[str], output_dir: Path) -> dict:
    """Stage 2: Log-Mel spectrogram extraction."""
    from src.features.spectrogram import extract_spectrograms_batch

    logger.info("=" * 60)
    logger.info("STAGE 2: Spectrogram Extraction")
    logger.info("=" * 60)
    start = time.time()
    stats = extract_spectrograms_batch(audio_paths, output_dir, show_progress=True)
    logger.info(f"Spectrogram stage completed in {time.time() - start:.1f}s")
    return stats


def run_prosodic_stage(audio_paths: list[str], output_dir: Path) -> dict:
    """Stage 3: Prosodic contour extraction."""
    from src.features.prosodic import extract_prosodics_batch

    logger.info("=" * 60)
    logger.info("STAGE 3: Prosodic Contour Extraction")
    logger.info("=" * 60)
    start = time.time()
    stats = extract_prosodics_batch(audio_paths, output_dir, show_progress=True)
    logger.info(f"Prosodic stage completed in {time.time() - start:.1f}s")
    return stats


def run_hubert_stage(audio_paths: list[str], output_dir: Path) -> dict:
    """Stage 4: HuBERT embedding extraction."""
    from src.features.hubert_extractor import extract_hubert_embeddings

    logger.info("=" * 60)
    logger.info("STAGE 4: HuBERT Embedding Extraction")
    logger.info("=" * 60)
    device = get_device()
    start = time.time()
    stats = extract_hubert_embeddings(
        audio_paths, output_dir, device=str(device), show_progress=True,
    )
    logger.info(f"HuBERT stage completed in {time.time() - start:.1f}s")
    return stats


def write_metadata_csv(
    records: list[dict],
    metadata_path: Path,
) -> None:
    """Write the master metadata CSV."""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "file_id", "dataset", "audio_path", "emotion_4class", "speaker_id",
        "spectrogram_path", "prosody_path", "hubert_path", "transcript_path",
        "duration_sec",
    ]

    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rec in records:
            file_id = rec["file_id"]
            duration = get_audio_duration(rec["audio_path"])

            writer.writerow({
                "file_id": file_id,
                "dataset": rec["dataset"],
                "audio_path": rec["audio_path"],
                "emotion_4class": rec["emotion_4class"],
                "speaker_id": rec["speaker_id"],
                "spectrogram_path": str(SPECTROGRAM_DIR / f"{file_id}.npy"),
                "prosody_path": str(PROSODIC_DIR / f"{file_id}.npy"),
                "hubert_path": str(HUBERT_DIR / f"{file_id}.npy"),
                "transcript_path": str(TRANSCRIPT_DIR / f"{file_id}.json"),
                "duration_sec": f"{duration:.3f}",
            })

    logger.info(f"Metadata written to {metadata_path} ({len(records)} rows)")


def main():
    parser = argparse.ArgumentParser(description="MERA: Run all feature extraction")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--datasets", type=str, default="iemocap,ravdess,cremad",
                        help="Comma-separated dataset names to process")
    parser.add_argument("--data-raw-dir", type=str, default=None,
                        help="Override raw data directory (default: data/raw)")
    parser.add_argument("--skip-whisper", action="store_true",
                        help="Skip Whisper transcription stage")
    parser.add_argument("--skip-spectrogram", action="store_true",
                        help="Skip spectrogram extraction stage")
    parser.add_argument("--skip-prosodic", action="store_true",
                        help="Skip prosodic contour extraction stage")
    parser.add_argument("--skip-hubert", action="store_true",
                        help="Skip HuBERT embedding extraction stage")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Only regenerate metadata.csv (skip all extraction)")
    args = parser.parse_args()

    # ---- Setup ----
    config_path = PROJECT_ROOT / args.config
    config = load_config(config_path)
    configure_mps_memory()
    seed_everything(config.seed)

    datasets = [d.strip() for d in args.datasets.split(",")]
    data_raw_dir = Path(args.data_raw_dir) if args.data_raw_dir else DEFAULT_DATA_ROOT / "raw"

    logger.info(f"Processing datasets: {datasets}")
    logger.info(f"Raw data directory: {data_raw_dir}")

    # ---- Collect audio records ----
    records = collect_audio_records(datasets, data_raw_dir)
    if not records:
        logger.error("No audio records found. Check dataset directories.")
        sys.exit(1)

    audio_paths = [rec["audio_path"] for rec in records]

    if not args.metadata_only:
        # ---- Stage 1: Whisper ----
        if not args.skip_whisper:
            run_whisper_stage(audio_paths, TRANSCRIPT_DIR, backend=config.whisper_backend)

        # ---- Stage 2: Spectrograms ----
        if not args.skip_spectrogram:
            run_spectrogram_stage(audio_paths, SPECTROGRAM_DIR)

        # ---- Stage 3: Prosodic Contours ----
        if not args.skip_prosodic:
            run_prosodic_stage(audio_paths, PROSODIC_DIR)

        # ---- Stage 4: HuBERT Embeddings ----
        if not args.skip_hubert:
            run_hubert_stage(audio_paths, HUBERT_DIR)

    # ---- Write metadata CSV ----
    write_metadata_csv(records, METADATA_PATH)

    logger.info("=" * 60)
    logger.info("ALL PREPROCESSING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
