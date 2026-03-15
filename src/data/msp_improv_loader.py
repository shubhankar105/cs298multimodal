"""MSP-IMPROV dataset loader.

Parses the MSP-IMPROV dataset from UT Dallas. This dataset contains
improvised emotional speech from 12 actors and is used as a held-out
validation set (not trained on).

Expected directory layout::

    msp_improv_root/
        session1/
            ...
        ...
        Labels/
            labels.csv   (or similar label file)
        Audios/ or audio/
            ...

The exact structure can vary by distribution. This loader supports
both a flat audio directory with a labels file, and a structured
directory with session subdirectories.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MSPImprovUtterance:
    """A single parsed MSP-IMPROV utterance with metadata."""

    utterance_id: str
    audio_path: str
    emotion: str
    emotion_4class: Optional[str]
    speaker_id: str
    session_id: str
    gender: str


MSP_EMOTION_MAP = {
    "A": "angry",
    "H": "happy",
    "S": "sad",
    "N": "neutral",
    "angry": "angry",
    "happy": "happy",
    "sad": "sad",
    "neutral": "neutral",
    "Angry": "angry",
    "Happy": "happy",
    "Sad": "sad",
    "Neutral": "neutral",
}

MSP_4CLASS_MAP = {
    "angry": "angry",
    "happy": "happy",
    "sad": "sad",
    "neutral": "neutral",
}


def _find_audio_dir(root_dir: Path) -> Optional[Path]:
    """Locate the audio directory within the MSP-IMPROV structure."""
    candidates = ["Audios", "audio", "Audio", "wav", "sentences"]
    for name in candidates:
        path = root_dir / name
        if path.exists():
            return path
    # Fallback: audio files might be directly in root
    if list(root_dir.glob("*.wav")):
        return root_dir
    return None


def _find_labels_file(root_dir: Path) -> Optional[Path]:
    """Locate the labels file within the MSP-IMPROV structure."""
    candidates = [
        root_dir / "Labels" / "labels.csv",
        root_dir / "labels.csv",
        root_dir / "Labels" / "labels.txt",
        root_dir / "labels.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _parse_labels_file(labels_path: Path) -> Dict[str, dict]:
    """Parse a labels CSV/TSV file into a dict keyed by utterance ID.

    Supports common formats with columns for utterance ID, emotion,
    and optional speaker/gender metadata.

    Returns:
        Dict mapping utterance_id → {emotion, speaker_id, gender, session_id}.
    """
    labels: Dict[str, dict] = {}

    with open(labels_path, "r") as f:
        # Try to detect delimiter
        sample = f.read(2048)
        f.seek(0)
        delimiter = "\t" if "\t" in sample else ","
        reader = csv.DictReader(f, delimiter=delimiter)

        # Normalize header names
        if reader.fieldnames is None:
            return labels

        # Map common header variations
        header_map: Dict[str, str] = {}
        for h in reader.fieldnames:
            h_lower = h.strip().lower()
            if h_lower in ("utteranceid", "utterance_id", "file", "filename", "id", "name"):
                header_map["utterance_id"] = h
            elif h_lower in ("emotion", "emo", "label", "emotionlabel"):
                header_map["emotion"] = h
            elif h_lower in ("speaker", "speaker_id", "speakerid", "actor"):
                header_map["speaker_id"] = h
            elif h_lower in ("gender", "sex"):
                header_map["gender"] = h
            elif h_lower in ("session", "session_id", "sessionid"):
                header_map["session_id"] = h

        if "utterance_id" not in header_map or "emotion" not in header_map:
            logger.warning(
                f"Labels file {labels_path} missing required columns "
                f"(utterance_id, emotion). Found headers: {reader.fieldnames}"
            )
            return labels

        for row in reader:
            utt_id = row[header_map["utterance_id"]].strip()
            emotion_raw = row[header_map["emotion"]].strip()

            emotion = MSP_EMOTION_MAP.get(emotion_raw)
            if emotion is None:
                continue

            labels[utt_id] = {
                "emotion": emotion,
                "speaker_id": row.get(header_map.get("speaker_id", ""), "unknown").strip()
                if "speaker_id" in header_map
                else "unknown",
                "gender": row.get(header_map.get("gender", ""), "unknown").strip()
                if "gender" in header_map
                else "unknown",
                "session_id": row.get(header_map.get("session_id", ""), "unknown").strip()
                if "session_id" in header_map
                else "unknown",
            }

    return labels


def load_msp_improv(
    root_dir: str | Path,
    four_class: bool = True,
) -> List[MSPImprovUtterance]:
    """Load MSP-IMPROV utterances from the dataset directory.

    Args:
        root_dir: Root directory of the MSP-IMPROV dataset.
        four_class: If True, filter to 4-class emotions only.

    Returns:
        List of MSPImprovUtterance records.
    """
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"MSP-IMPROV root directory not found: {root_dir}")

    labels_path = _find_labels_file(root_dir)
    if labels_path is None:
        raise FileNotFoundError(
            f"No labels file found in {root_dir}. "
            f"Expected labels.csv in root or Labels/ subdirectory."
        )

    audio_dir = _find_audio_dir(root_dir)
    if audio_dir is None:
        raise FileNotFoundError(
            f"No audio directory found in {root_dir}. "
            f"Expected Audios/ or audio/ subdirectory."
        )

    labels = _parse_labels_file(labels_path)
    if not labels:
        logger.warning(f"No valid labels parsed from {labels_path}")
        return []

    utterances: List[MSPImprovUtterance] = []

    # Match audio files to labels
    for wav_file in sorted(audio_dir.rglob("*.wav")):
        utt_id = wav_file.stem
        if utt_id not in labels:
            continue

        meta = labels[utt_id]
        emotion = meta["emotion"]
        emotion_4class = MSP_4CLASS_MAP.get(emotion)

        if four_class and emotion_4class is None:
            continue

        utterances.append(
            MSPImprovUtterance(
                utterance_id=f"msp_{utt_id}",
                audio_path=str(wav_file),
                emotion=emotion,
                emotion_4class=emotion_4class,
                speaker_id=f"msp_{meta['speaker_id']}",
                session_id=meta["session_id"],
                gender=meta["gender"],
            )
        )

    logger.info(f"Loaded {len(utterances)} MSP-IMPROV utterances from {root_dir}")
    return utterances
