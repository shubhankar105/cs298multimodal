"""Custom dataset and collate functions for variable-length audio features.

Two dataset modes:
1. **CachedFeatureDataset** — loads pre-computed features from disk using
   memory-mapped NumPy arrays (``mmap_mode='r'``) so that only the data
   actually accessed is paged into RAM.
2. Collation pads all sequences in a batch to the longest and returns
   boolean masks so attention layers can ignore padded positions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

EMOTION_TO_IDX = {"angry": 0, "happy": 1, "sad": 2, "neutral": 3}
IDX_TO_EMOTION = {v: k for k, v in EMOTION_TO_IDX.items()}


class CachedFeatureDataset(Dataset):
    """Loads pre-computed features from disk (spectrograms, prosody, HuBERT, transcripts).

    Each sample returns a dict with:
    - ``spectrogram``:  ``torch.Tensor`` of shape ``(128, T)``
    - ``prosody``:      ``torch.Tensor`` of shape ``(10, T)``
    - ``hubert``:       ``torch.Tensor`` of shape ``(25, T_h, 1024)``
    - ``transcript``:   ``str``
    - ``emotion``:      ``int`` (0–3)
    - ``file_id``:      ``str``
    - ``speaker_id``:   ``str``

    Memory efficiency:
    - HuBERT embeddings are memory-mapped (``np.load(…, mmap_mode='r')``)
      and cast to float32 only for the slice that is returned.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        augmentor: Optional[object] = None,
        spec_augment: Optional[object] = None,
    ):
        """
        Args:
            metadata_df: DataFrame with columns ``file_id``, ``emotion_4class``,
                ``speaker_id``, ``spectrogram_path``, ``prosody_path``,
                ``hubert_path``, ``transcript_path``.
            augmentor: Optional AudioAugmentor (unused here since we work with
                cached features, but kept for interface parity).
            spec_augment: Optional SpecAugment applied to spectrograms.
        """
        self.metadata = metadata_df.reset_index(drop=True)
        self.augmentor = augmentor
        self.spec_augment = spec_augment

        # Validate required columns
        required = {
            "file_id", "emotion_4class", "speaker_id",
            "spectrogram_path", "prosody_path", "hubert_path", "transcript_path",
        }
        missing = required - set(self.metadata.columns)
        if missing:
            raise ValueError(f"metadata_df is missing columns: {missing}")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        row = self.metadata.iloc[idx]

        # Load spectrogram (float32, small — no mmap needed)
        spec = np.load(row["spectrogram_path"]).astype(np.float32)

        # Load prosodic contours (float32, small)
        prosody = np.load(row["prosody_path"]).astype(np.float32)

        # Load HuBERT (float16 on disk → float32 in memory, memory-mapped)
        hubert = np.load(row["hubert_path"], mmap_mode="r")
        hubert = np.array(hubert, dtype=np.float32)  # materialise slice

        # Load transcript
        transcript = ""
        transcript_path = Path(row["transcript_path"])
        if transcript_path.exists():
            with open(transcript_path, "r") as f:
                transcript_data = json.load(f)
            transcript = transcript_data.get("text", "")

        # SpecAugment (training-time only)
        if self.spec_augment is not None:
            spec = self.spec_augment(spec)

        return {
            "spectrogram": torch.from_numpy(spec),
            "prosody": torch.from_numpy(prosody),
            "hubert": torch.from_numpy(hubert),
            "transcript": transcript,
            "emotion": EMOTION_TO_IDX[row["emotion_4class"]],
            "file_id": row["file_id"],
            "speaker_id": row["speaker_id"],
        }


def collate_features(batch: list) -> dict:
    """Collate variable-length feature dicts into a padded batch.

    Returns a dict with padded tensors and boolean masks (``True`` for
    real positions, ``False`` for padding).

    Keys returned:
    - ``spectrogram``:       ``(B, 128, T_max)``
    - ``spectrogram_mask``:  ``(B, T_max)``   bool
    - ``prosody``:           ``(B, 10, T_max)``
    - ``prosody_mask``:      ``(B, T_max)``   bool
    - ``hubert``:            ``(B, 25, T_h_max, 1024)``
    - ``hubert_mask``:       ``(B, T_h_max)`` bool
    - ``emotion``:           ``(B,)``         long
    - ``transcript``:        ``List[str]``
    - ``file_id``:           ``List[str]``
    - ``speaker_id``:        ``List[str]``
    """
    batch_size = len(batch)

    # Determine max lengths
    max_spec_frames = max(b["spectrogram"].shape[1] for b in batch)
    max_pros_frames = max(b["prosody"].shape[1] for b in batch)
    max_hub_frames = max(b["hubert"].shape[1] for b in batch)

    n_mels = batch[0]["spectrogram"].shape[0]  # 128
    n_pros = batch[0]["prosody"].shape[0]       # 10
    n_layers = batch[0]["hubert"].shape[0]      # 25
    hub_dim = batch[0]["hubert"].shape[2]       # 1024

    # Allocate padded tensors (zeros = silence/padding)
    specs = torch.zeros(batch_size, n_mels, max_spec_frames)
    spec_masks = torch.zeros(batch_size, max_spec_frames, dtype=torch.bool)

    prosodies = torch.zeros(batch_size, n_pros, max_pros_frames)
    pros_masks = torch.zeros(batch_size, max_pros_frames, dtype=torch.bool)

    huberts = torch.zeros(batch_size, n_layers, max_hub_frames, hub_dim)
    hub_masks = torch.zeros(batch_size, max_hub_frames, dtype=torch.bool)

    emotions = torch.zeros(batch_size, dtype=torch.long)
    transcripts: list[str] = []
    file_ids: list[str] = []
    speaker_ids: list[str] = []

    for i, b in enumerate(batch):
        # Spectrogram
        t_s = b["spectrogram"].shape[1]
        specs[i, :, :t_s] = b["spectrogram"]
        spec_masks[i, :t_s] = True

        # Prosody
        t_p = b["prosody"].shape[1]
        prosodies[i, :, :t_p] = b["prosody"]
        pros_masks[i, :t_p] = True

        # HuBERT
        t_h = b["hubert"].shape[1]
        huberts[i, :, :t_h, :] = b["hubert"]
        hub_masks[i, :t_h] = True

        emotions[i] = b["emotion"]
        transcripts.append(b["transcript"])
        file_ids.append(b["file_id"])
        speaker_ids.append(b["speaker_id"])

    return {
        "spectrogram": specs,
        "spectrogram_mask": spec_masks,
        "prosody": prosodies,
        "prosody_mask": pros_masks,
        "hubert": huberts,
        "hubert_mask": hub_masks,
        "emotion": emotions,
        "transcript": transcripts,
        "file_id": file_ids,
        "speaker_id": speaker_ids,
    }
