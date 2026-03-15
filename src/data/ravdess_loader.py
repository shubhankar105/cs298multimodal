"""RAVDESS dataset loader.

Parses the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).
Labels are encoded entirely in the filename structure.

Filename format: ``03-01-06-01-02-01-12.wav``
    Position 1: Modality (01=full-AV, 02=video-only, 03=audio-only)
    Position 2: Vocal channel (01=speech, 02=song)
    Position 3: Emotion (01–08)
    Position 4: Intensity (01=normal, 02=strong)
    Position 5: Statement (01 or 02)
    Position 6: Repetition (01 or 02)
    Position 7: Actor ID (01–24, odd=male, even=female)

Expected directory layout::

    ravdess_root/
        Actor_01/
            03-01-01-01-01-01-01.wav
            ...
        Actor_02/
            ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RAVDESSUtterance:
    """A single parsed RAVDESS utterance with metadata."""

    utterance_id: str
    audio_path: str
    emotion: str
    emotion_4class: Optional[str]
    intensity: str
    statement: str
    repetition: str
    actor_id: int
    gender: str
    speaker_id: str


RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}

RAVDESS_4CLASS_MAP = {
    "neutral": "neutral",
    "calm": "neutral",       # Merge calm → neutral
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fear": None,            # Drop for 4-class
    "disgust": None,         # Drop for 4-class
    "surprise": None,        # Drop for 4-class
}


def parse_ravdess_filename(filepath: str | Path) -> Optional[dict]:
    """Parse a RAVDESS filename into metadata fields.

    Args:
        filepath: Path to a RAVDESS audio file.

    Returns:
        Dict with parsed metadata, or None if the filename is not valid RAVDESS format.
    """
    filepath = Path(filepath)
    stem = filepath.stem
    parts = stem.split("-")

    if len(parts) != 7:
        logger.debug(f"Skipping non-RAVDESS file: {filepath.name}")
        return None

    try:
        actor_id = int(parts[6])
    except ValueError:
        return None

    emotion_code = parts[2]
    if emotion_code not in RAVDESS_EMOTION_MAP:
        return None

    emotion = RAVDESS_EMOTION_MAP[emotion_code]
    emotion_4class = RAVDESS_4CLASS_MAP.get(emotion)

    return {
        "modality": parts[0],
        "vocal_channel": parts[1],
        "emotion": emotion,
        "emotion_4class": emotion_4class,
        "intensity": "normal" if parts[3] == "01" else "strong",
        "statement": parts[4],
        "repetition": parts[5],
        "actor_id": actor_id,
        "gender": "male" if actor_id % 2 == 1 else "female",
    }


def load_ravdess(
    root_dir: str | Path,
    audio_only: bool = True,
    speech_only: bool = True,
    four_class: bool = True,
) -> List[RAVDESSUtterance]:
    """Load all RAVDESS utterances from the dataset directory.

    Args:
        root_dir: Root directory containing Actor_XX subdirectories.
        audio_only: If True, only load audio-only files (modality=03).
        speech_only: If True, only load speech files (vocal_channel=01).
        four_class: If True, filter to only emotions mappable to the 4-class scheme.

    Returns:
        List of RAVDESSUtterance records.
    """
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"RAVDESS root directory not found: {root_dir}")

    utterances: List[RAVDESSUtterance] = []

    for wav_file in sorted(root_dir.rglob("*.wav")):
        meta = parse_ravdess_filename(wav_file)
        if meta is None:
            continue

        # Filter by modality
        if audio_only and meta["modality"] != "03":
            continue

        # Filter by vocal channel
        if speech_only and meta["vocal_channel"] != "01":
            continue

        # Filter by 4-class compatibility
        if four_class and meta["emotion_4class"] is None:
            continue

        utterance_id = f"ravdess_{wav_file.stem}"
        speaker_id = f"ravdess_actor{meta['actor_id']:02d}"

        utterances.append(
            RAVDESSUtterance(
                utterance_id=utterance_id,
                audio_path=str(wav_file),
                emotion=meta["emotion"],
                emotion_4class=meta["emotion_4class"],
                intensity=meta["intensity"],
                statement=meta["statement"],
                repetition=meta["repetition"],
                actor_id=meta["actor_id"],
                gender=meta["gender"],
                speaker_id=speaker_id,
            )
        )

    logger.info(f"Loaded {len(utterances)} RAVDESS utterances from {root_dir}")
    return utterances
