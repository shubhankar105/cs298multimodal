"""Unified dataset registry for MERA.

Provides a single interface to load any supported dataset by name,
returning standardized records with a common schema regardless of
the source dataset's format.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EmotionRecord:
    """Standardized record returned by all dataset loaders.

    This common schema enables uniform processing regardless of the
    source dataset.
    """

    file_id: str
    dataset: str
    audio_path: Optional[str]
    text: Optional[str]
    emotion_4class: str
    emotion_fine: str
    speaker_id: str
    gender: str
    session_id: str
    duration_sec: Optional[float] = None


# Canonical 4-class emotion labels used throughout MERA
EMOTION_LABELS_4CLASS = ["angry", "happy", "sad", "neutral"]
EMOTION_TO_IDX = {label: idx for idx, label in enumerate(EMOTION_LABELS_4CLASS)}
IDX_TO_EMOTION = {idx: label for idx, label in enumerate(EMOTION_LABELS_4CLASS)}

# Supported dataset names
SUPPORTED_DATASETS = {"iemocap", "ravdess", "cremad", "msp_improv", "goemotions"}


def load_dataset_by_name(
    name: str,
    root_dir: str | Path,
    four_class: bool = True,
    **kwargs,
) -> List[EmotionRecord]:
    """Load a dataset by name and return standardized EmotionRecords.

    Args:
        name: Dataset name. One of: iemocap, ravdess, cremad, msp_improv, goemotions.
        root_dir: Root directory for the dataset files.
        four_class: If True, filter to 4-class emotions only.
        **kwargs: Additional keyword arguments passed to the specific loader.

    Returns:
        List of EmotionRecord instances with standardized fields.

    Raises:
        ValueError: If the dataset name is not recognized.
        FileNotFoundError: If the dataset directory does not exist.
    """
    name = name.lower().strip()
    if name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unknown dataset: '{name}'. Supported: {sorted(SUPPORTED_DATASETS)}"
        )

    loader_fn = _LOADERS[name]
    return loader_fn(root_dir, four_class=four_class, **kwargs)


def _load_iemocap(root_dir: str | Path, four_class: bool = True, **kwargs) -> List[EmotionRecord]:
    from src.data.iemocap_loader import load_iemocap

    sessions = kwargs.get("sessions", None)
    utterances = load_iemocap(root_dir, sessions=sessions)

    records = []
    for u in utterances:
        records.append(
            EmotionRecord(
                file_id=f"iemocap_{u.utterance_id}",
                dataset="iemocap",
                audio_path=u.audio_path,
                text=None,
                emotion_4class=u.emotion,
                emotion_fine=u.emotion,
                speaker_id=u.speaker_id,
                gender=u.gender,
                session_id=str(u.session_id),
            )
        )
    return records


def _load_ravdess(root_dir: str | Path, four_class: bool = True, **kwargs) -> List[EmotionRecord]:
    from src.data.ravdess_loader import load_ravdess

    utterances = load_ravdess(root_dir, four_class=four_class, **kwargs)

    records = []
    for u in utterances:
        records.append(
            EmotionRecord(
                file_id=u.utterance_id,
                dataset="ravdess",
                audio_path=u.audio_path,
                text=None,
                emotion_4class=u.emotion_4class or u.emotion,
                emotion_fine=u.emotion,
                speaker_id=u.speaker_id,
                gender=u.gender,
                session_id=f"actor{u.actor_id:02d}",
            )
        )
    return records


def _load_cremad(root_dir: str | Path, four_class: bool = True, **kwargs) -> List[EmotionRecord]:
    from src.data.cremad_loader import load_cremad

    utterances = load_cremad(root_dir, four_class=four_class)

    records = []
    for u in utterances:
        records.append(
            EmotionRecord(
                file_id=u.utterance_id,
                dataset="cremad",
                audio_path=u.audio_path,
                text=None,
                emotion_4class=u.emotion_4class or u.emotion,
                emotion_fine=u.emotion,
                speaker_id=u.speaker_id,
                gender="unknown",
                session_id=f"actor{u.actor_id}",
            )
        )
    return records


def _load_msp_improv(root_dir: str | Path, four_class: bool = True, **kwargs) -> List[EmotionRecord]:
    from src.data.msp_improv_loader import load_msp_improv

    utterances = load_msp_improv(root_dir, four_class=four_class)

    records = []
    for u in utterances:
        records.append(
            EmotionRecord(
                file_id=u.utterance_id,
                dataset="msp_improv",
                audio_path=u.audio_path,
                text=None,
                emotion_4class=u.emotion_4class or u.emotion,
                emotion_fine=u.emotion,
                speaker_id=u.speaker_id,
                gender=u.gender,
                session_id=u.session_id,
            )
        )
    return records


def _load_goemotions(root_dir: str | Path, four_class: bool = True, **kwargs) -> List[EmotionRecord]:
    from src.data.goemotions_loader import load_goemotions_from_local

    split = kwargs.get("split", "train")
    records_raw = load_goemotions_from_local(str(root_dir), split=split, four_class=four_class)

    records = []
    for r in records_raw:
        records.append(
            EmotionRecord(
                file_id=r.record_id,
                dataset="goemotions",
                audio_path=None,
                text=r.text,
                emotion_4class=r.emotion_4class or "",
                emotion_fine=",".join(r.fine_labels),
                speaker_id="unknown",
                gender="unknown",
                session_id="unknown",
            )
        )
    return records


_LOADERS = {
    "iemocap": _load_iemocap,
    "ravdess": _load_ravdess,
    "cremad": _load_cremad,
    "msp_improv": _load_msp_improv,
    "goemotions": _load_goemotions,
}


def get_class_distribution(records: List[EmotionRecord]) -> dict:
    """Compute the emotion class distribution of a record list.

    Args:
        records: List of EmotionRecord instances.

    Returns:
        Dict mapping emotion label → count.
    """
    from collections import Counter

    return dict(Counter(r.emotion_4class for r in records))


def records_to_dataframe(records: List[EmotionRecord]):
    """Convert EmotionRecords to a pandas DataFrame.

    Args:
        records: List of EmotionRecord instances.

    Returns:
        pandas.DataFrame with columns matching EmotionRecord fields.
    """
    import dataclasses
    import pandas as pd

    rows = [dataclasses.asdict(r) for r in records]
    return pd.DataFrame(rows)
