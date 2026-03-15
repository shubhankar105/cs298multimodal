"""HuBERT-Large layer-wise embedding extraction.

Extracts hidden states from **all 25 layers** (input CNN + 24 transformer
layers) of a frozen HuBERT-Large model.  Embeddings are cached to disk as
float16 NumPy arrays to manage the ~20–50 GB cache for IEMOCAP.

Memory strategy (24 GB unified memory on M5 Pro):
- Model footprint: ~1.2 GB
- Process utterances **sequentially** (batch_size=1)
- Call ``torch.mps.empty_cache()`` every *cache_clear_interval* utterances
- Save immediately after each utterance to minimise peak RAM

Output per utterance: ``np.ndarray`` of shape ``(25, T, 1024)`` where
- 25 = layers (input features + 24 transformer blocks)
- T  = time frames (~50 frames/s for HuBERT)
- 1024 = hidden dim of HuBERT-Large

Saved as **float16** to halve disk usage.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from src.data.audio_utils import load_and_preprocess

logger = logging.getLogger(__name__)

MODEL_NAME = "facebook/hubert-large-ls960-ft"
HUBERT_SR = 16000
MAX_DURATION_SAMPLES = HUBERT_SR * 15  # 15 s


def extract_hubert_embeddings(
    audio_paths: List[str | Path],
    output_dir: str | Path,
    device: str | torch.device = "mps",
    model_name: str = MODEL_NAME,
    max_duration_samples: int = MAX_DURATION_SAMPLES,
    cache_clear_interval: int = 50,
    show_progress: bool = True,
) -> dict:
    """Extract and cache HuBERT layer-wise embeddings for a list of audio files.

    Args:
        audio_paths: List of audio file paths.
        output_dir: Directory for ``.npy`` output files.
        device: Torch device string (``"mps"``, ``"cpu"``, ``"cuda"``).
        model_name: HuggingFace model identifier.
        max_duration_samples: Truncate audio longer than this many samples.
        cache_clear_interval: Call ``torch.mps.empty_cache()`` every *N* files.
        show_progress: Display tqdm progress bar.

    Returns:
        Dict with ``processed``, ``skipped``, ``errors`` counts.
    """
    from transformers import HubertModel, Wav2Vec2FeatureExtractor
    from tqdm import tqdm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device) if isinstance(device, str) else device

    # ---- Load model (frozen, no gradients) ----
    logger.info(f"Loading HuBERT model: {model_name}")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = HubertModel.from_pretrained(model_name, output_hidden_states=True)
    model = model.to(device)
    model.eval()
    logger.info(f"HuBERT loaded on {device}")

    stats = {"processed": 0, "skipped": 0, "errors": []}
    iterator = tqdm(audio_paths, desc="HuBERT embeddings", disable=not show_progress)

    with torch.no_grad():
        for idx, audio_path in enumerate(iterator):
            file_id = Path(audio_path).stem
            output_path = output_dir / f"{file_id}.npy"

            if output_path.exists():
                stats["skipped"] += 1
                continue

            try:
                # Load & preprocess
                audio = load_and_preprocess(audio_path, sr=HUBERT_SR)
                if len(audio) > max_duration_samples:
                    audio = audio[:max_duration_samples]

                # Feature extraction
                inputs = feature_extractor(
                    audio,
                    sampling_rate=HUBERT_SR,
                    return_tensors="pt",
                    padding=False,
                )
                input_values = inputs.input_values.to(device)

                # Forward pass
                outputs = model(input_values)

                # Collect all hidden states: tuple of 25 tensors, each (1, T, 1024)
                hidden_states = outputs.hidden_states

                # Stack → (25, T, 1024)
                stacked = torch.stack(hidden_states, dim=0).squeeze(1)

                # Save as float16
                embeddings = stacked.cpu().numpy().astype(np.float16)
                np.save(str(output_path), embeddings)
                stats["processed"] += 1

            except Exception as e:
                logger.error(f"Failed HuBERT extraction for {file_id}: {e}")
                stats["errors"].append(str(audio_path))

            # Periodic MPS cache clearing
            if (idx + 1) % cache_clear_interval == 0:
                _clear_cache(device)

    # Final cleanup
    del model
    _clear_cache(device)

    logger.info(
        f"HuBERT extraction: {stats['processed']} processed, "
        f"{stats['skipped']} skipped, {len(stats['errors'])} errors"
    )
    return stats


def load_hubert_embedding(
    path: str | Path,
    dtype: np.dtype = np.float32,
    mmap: bool = True,
) -> np.ndarray:
    """Load a cached HuBERT embedding from disk.

    Args:
        path: Path to the ``.npy`` file.
        dtype: Desired output dtype (float16 on disk → cast to this).
        mmap: If True, memory-map the file to avoid loading into RAM.

    Returns:
        ``np.ndarray`` of shape ``(25, T, 1024)``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HuBERT embedding not found: {path}")

    mmap_mode = "r" if mmap else None
    data = np.load(str(path), mmap_mode=mmap_mode)

    if dtype != data.dtype and not mmap:
        data = data.astype(dtype)

    return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clear_cache(device: torch.device) -> None:
    """Clear the GPU / MPS cache to reclaim memory."""
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
