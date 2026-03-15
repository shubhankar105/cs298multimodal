"""CREMA-D dataset loader.

Parses the Crowd-sourced Emotional Multimodal Actors Dataset (CREMA-D).
Labels are encoded in the filename.

Filename format: ``1001_DFA_ANG_XX.wav``
    Position 1: Actor ID (1001–1091)
    Position 2: Sentence code (DFA, IEO, IOM, ITS, ITH, MTI, TAI, TIE, TSI, WSI, IWL, IWW)
    Position 3: Emotion (ANG, DIS, FEA, HAP, NEU, SAD)
    Position 4: Intensity (XX=unspecified, LO=low, MD=medium, HI=high)

Expected directory layout::

    cremad_root/
        AudioWAV/
            1001_DFA_ANG_XX.wav
            ...
        (or .wav files directly in root)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CREMADUtterance:
    """A single parsed CREMA-D utterance with metadata."""

    utterance_id: str
    audio_path: str
    emotion: str
    emotion_4class: Optional[str]
    intensity: str
    sentence_code: str
    actor_id: int
    speaker_id: str


CREMAD_EMOTION_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

CREMAD_4CLASS_MAP = {
    "angry": "angry",
    "happy": "happy",
    "sad": "sad",
    "neutral": "neutral",
    "disgust": None,
    "fear": None,
}

CREMAD_INTENSITY_MAP = {
    "XX": "unspecified",
    "LO": "low",
    "MD": "medium",
    "HI": "high",
}


def parse_cremad_filename(filepath: str | Path) -> Optional[dict]:
    """Parse a CREMA-D filename into metadata fields.

    Args:
        filepath: Path to a CREMA-D audio file.

    Returns:
        Dict with parsed metadata, or None if the filename is not valid.
    """
    filepath = Path(filepath)
    stem = filepath.stem
    parts = stem.split("_")

    if len(parts) != 4:
        logger.debug(f"Skipping non-CREMA-D file: {filepath.name}")
        return None

    actor_str, sentence_code, emotion_code, intensity_code = parts

    try:
        actor_id = int(actor_str)
    except ValueError:
        return None

    if emotion_code not in CREMAD_EMOTION_MAP:
        return None

    emotion = CREMAD_EMOTION_MAP[emotion_code]
    emotion_4class = CREMAD_4CLASS_MAP.get(emotion)
    intensity = CREMAD_INTENSITY_MAP.get(intensity_code, "unknown")

    return {
        "actor_id": actor_id,
        "sentence_code": sentence_code,
        "emotion": emotion,
        "emotion_4class": emotion_4class,
        "intensity": intensity,
    }


def load_cremad(
    root_dir: str | Path,
    four_class: bool = True,
) -> List[CREMADUtterance]:
    """Load all CREMA-D utterances from the dataset directory.

    Args:
        root_dir: Root directory containing the audio files (or an AudioWAV
            subdirectory).
        four_class: If True, filter to only emotions mappable to the 4-class scheme.

    Returns:
        List of CREMADUtterance records.
    """
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"CREMA-D root directory not found: {root_dir}")

    # CREMA-D files may be in an AudioWAV subdirectory or directly in root
    audio_dir = root_dir / "AudioWAV"
    if not audio_dir.exists():
        audio_dir = root_dir

    utterances: List[CREMADUtterance] = []

    for wav_file in sorted(audio_dir.rglob("*.wav")):
        meta = parse_cremad_filename(wav_file)
        if meta is None:
            continue

        if four_class and meta["emotion_4class"] is None:
            continue

        utterance_id = f"cremad_{wav_file.stem}"
        speaker_id = f"cremad_actor{meta['actor_id']}"

        utterances.append(
            CREMADUtterance(
                utterance_id=utterance_id,
                audio_path=str(wav_file),
                emotion=meta["emotion"],
                emotion_4class=meta["emotion_4class"],
                intensity=meta["intensity"],
                sentence_code=meta["sentence_code"],
                actor_id=meta["actor_id"],
                speaker_id=speaker_id,
            )
        )

    logger.info(f"Loaded {len(utterances)} CREMA-D utterances from {root_dir}")
    return utterances
