"""GoEmotions dataset loader.

Loads the GoEmotions text-only dataset (58K Reddit comments, 27 emotion labels)
from HuggingFace datasets for pre-training the DeBERTa text emotion classifier.

The 27 fine-grained emotions are mapped to 4 classes for domain adaptation:
    Happy:   admiration, amusement, approval, caring, desire, excitement,
             gratitude, joy, love, optimism, pride, relief
    Angry:   anger, annoyance, disapproval, disgust
    Sad:     disappointment, embarrassment, grief, remorse, sadness
    Neutral: neutral, confusion, curiosity, realization, surprise
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GoEmotionsRecord:
    """A single GoEmotions text record."""

    record_id: str
    text: str
    fine_labels: List[str]
    emotion_4class: Optional[str]


# GoEmotions label index → label name (from the HuggingFace dataset)
GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]

# Mapping from fine-grained GoEmotions labels to 4-class scheme
GOEMOTIONS_4CLASS_MAP = {
    # Happy
    "admiration": "happy",
    "amusement": "happy",
    "approval": "happy",
    "caring": "happy",
    "desire": "happy",
    "excitement": "happy",
    "gratitude": "happy",
    "joy": "happy",
    "love": "happy",
    "optimism": "happy",
    "pride": "happy",
    "relief": "happy",
    # Angry
    "anger": "angry",
    "annoyance": "angry",
    "disapproval": "angry",
    "disgust": "angry",
    # Sad
    "disappointment": "sad",
    "embarrassment": "sad",
    "grief": "sad",
    "remorse": "sad",
    "sadness": "sad",
    # Neutral
    "neutral": "neutral",
    "confusion": "neutral",
    "curiosity": "neutral",
    "realization": "neutral",
    "surprise": "neutral",
    # Unmapped (dropped for 4-class)
    "fear": None,
    "nervousness": None,
}


def _resolve_4class(fine_labels: List[str]) -> Optional[str]:
    """Resolve a list of fine-grained labels to a single 4-class label.

    Uses majority vote among the mapped labels. If there's a tie or no
    valid labels, returns None.

    Args:
        fine_labels: List of fine-grained GoEmotions label strings.

    Returns:
        The 4-class label, or None if no valid mapping exists.
    """
    from collections import Counter

    mapped = [
        GOEMOTIONS_4CLASS_MAP[label]
        for label in fine_labels
        if label in GOEMOTIONS_4CLASS_MAP and GOEMOTIONS_4CLASS_MAP[label] is not None
    ]
    if not mapped:
        return None

    counts = Counter(mapped)
    winner, count = counts.most_common(1)[0]
    return winner


def load_goemotions(
    split: str = "train",
    four_class: bool = True,
    cache_dir: Optional[str] = None,
) -> List[GoEmotionsRecord]:
    """Load GoEmotions records from HuggingFace datasets.

    Args:
        split: Dataset split to load ("train", "validation", "test").
        four_class: If True, map to 4-class emotions and filter out unmappable.
        cache_dir: Optional cache directory for HuggingFace downloads.

    Returns:
        List of GoEmotionsRecord instances.

    Raises:
        ImportError: If the ``datasets`` package is not installed.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for GoEmotions. "
            "Install it with: pip install datasets"
        )

    logger.info(f"Loading GoEmotions split={split} from HuggingFace...")
    dataset = load_dataset(
        "google-research-datasets/go_emotions",
        "simplified",
        split=split,
        cache_dir=cache_dir,
    )

    records: List[GoEmotionsRecord] = []

    for idx, example in enumerate(dataset):
        text = example["text"]
        label_indices = example["labels"]
        fine_labels = [GOEMOTIONS_LABELS[i] for i in label_indices if i < len(GOEMOTIONS_LABELS)]

        emotion_4class = _resolve_4class(fine_labels) if four_class else None

        if four_class and emotion_4class is None:
            continue

        records.append(
            GoEmotionsRecord(
                record_id=f"goemotions_{split}_{idx}",
                text=text,
                fine_labels=fine_labels,
                emotion_4class=emotion_4class,
            )
        )

    logger.info(f"Loaded {len(records)} GoEmotions records (split={split})")
    return records


def load_goemotions_from_local(
    data_dir: str,
    split: str = "train",
    four_class: bool = True,
) -> List[GoEmotionsRecord]:
    """Load GoEmotions from local CSV files (offline fallback).

    Expects files named ``{split}.tsv`` in the given directory, with columns:
    text, labels (comma-separated indices), id.

    Args:
        data_dir: Directory containing the TSV files.
        split: Which split to load.
        four_class: If True, map to 4-class and filter.

    Returns:
        List of GoEmotionsRecord instances.
    """
    import csv
    from pathlib import Path

    data_path = Path(data_dir) / f"{split}.tsv"
    if not data_path.exists():
        raise FileNotFoundError(f"GoEmotions local file not found: {data_path}")

    records: List[GoEmotionsRecord] = []

    with open(data_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for idx, row in enumerate(reader):
            text = row.get("text", "")
            label_str = row.get("labels", "")

            try:
                label_indices = [int(x) for x in label_str.split(",") if x.strip()]
            except ValueError:
                continue

            fine_labels = [
                GOEMOTIONS_LABELS[i] for i in label_indices if i < len(GOEMOTIONS_LABELS)
            ]
            emotion_4class = _resolve_4class(fine_labels) if four_class else None

            if four_class and emotion_4class is None:
                continue

            records.append(
                GoEmotionsRecord(
                    record_id=f"goemotions_{split}_{idx}",
                    text=text,
                    fine_labels=fine_labels,
                    emotion_4class=emotion_4class,
                )
            )

    logger.info(f"Loaded {len(records)} GoEmotions records from local (split={split})")
    return records
