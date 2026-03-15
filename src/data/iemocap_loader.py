"""IEMOCAP dataset loader.

Parses the Interactive Emotional Dyadic Motion Capture (IEMOCAP) dataset.
Handles the dialog-based directory structure, label file parsing with regex,
and the standard 4-class emotion mapping (merging happy + excited).

Expected directory layout::

    iemocap_root/
        Session1/
            dialog/
                EmoEvaluation/
                    Ses01F_impro01.txt
                    ...
            sentences/
                wav/
                    Ses01F_impro01/
                        Ses01F_impro01_F000.wav
                        ...
        Session2/
            ...
        Session5/
            ...
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class IEMOCAPUtterance:
    """A single parsed IEMOCAP utterance with metadata."""

    utterance_id: str
    audio_path: str
    emotion: str
    valence: float
    arousal: float
    dominance: float
    speaker_id: str       # e.g. "Ses01F" or "Ses01M"
    session_id: int        # 1–5
    gender: str            # "M" or "F"
    dialog_type: str       # "scripted" or "improvised"


# Standard 4-class mapping used across IEMOCAP literature
EMOTION_MAP_4CLASS = {
    "ang": "angry",
    "hap": "happy",
    "exc": "happy",      # Merge excited into happy (standard practice)
    "sad": "sad",
    "neu": "neutral",
}

VALID_EMOTIONS = set(EMOTION_MAP_4CLASS.keys())

# Regex for parsing IEMOCAP EmoEvaluation label lines
# Matches: [start - end] utterance_id emotion [valence, arousal, dominance]
_LABEL_PATTERN = re.compile(
    r"\[(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\]\s+(\S+)\s+(\S+)\s+\[(\S+,\s*\S+,\s*\S+)\]"
)


def parse_iemocap_session(session_dir: str | Path) -> List[IEMOCAPUtterance]:
    """Parse a single IEMOCAP session directory into utterance records.

    Args:
        session_dir: Path to a session directory (e.g. ``data/raw/iemocap/Session1``).

    Returns:
        List of IEMOCAPUtterance dataclass instances for all valid 4-class utterances.

    Raises:
        FileNotFoundError: If the session directory or expected subdirectories
            do not exist.
    """
    session_dir = Path(session_dir)
    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")

    eval_dir = session_dir / "dialog" / "EmoEvaluation"
    wav_dir = session_dir / "sentences" / "wav"

    if not eval_dir.exists():
        raise FileNotFoundError(f"EmoEvaluation directory not found: {eval_dir}")
    if not wav_dir.exists():
        raise FileNotFoundError(f"WAV directory not found: {wav_dir}")

    utterances: List[IEMOCAPUtterance] = []

    for eval_file in sorted(eval_dir.glob("*.txt")):
        with open(eval_file, "r") as f:
            for line in f:
                match = _LABEL_PATTERN.search(line)
                if not match:
                    continue

                utt_id = match.group(3)
                emotion_raw = match.group(4)
                vad_str = match.group(5)

                if emotion_raw not in VALID_EMOTIONS:
                    continue

                # Parse valence, arousal, dominance
                try:
                    vad_parts = [v.strip() for v in vad_str.split(",")]
                    valence = float(vad_parts[0])
                    arousal = float(vad_parts[1])
                    dominance = float(vad_parts[2])
                except (ValueError, IndexError):
                    valence, arousal, dominance = 0.0, 0.0, 0.0

                # Derive metadata from utterance ID
                # e.g. Ses01F_impro01_F000 → parts = ["Ses01F", "impro01", "F000"]
                parts = utt_id.split("_")
                try:
                    session_id = int(parts[0][3:5])
                except (ValueError, IndexError):
                    logger.warning(f"Cannot parse session ID from {utt_id}, skipping")
                    continue

                # Gender from the last segment (e.g. "F000" → "F")
                gender = utt_id[-4] if len(utt_id) >= 4 else "U"
                speaker_id = f"Ses{session_id:02d}{gender}"
                dialog_type = "improvised" if "impro" in utt_id else "scripted"

                # Locate audio file
                dialog_id = "_".join(parts[:-1])
                audio_path = wav_dir / dialog_id / f"{utt_id}.wav"

                if not audio_path.exists():
                    logger.debug(f"Audio file not found: {audio_path}")
                    continue

                utterances.append(
                    IEMOCAPUtterance(
                        utterance_id=utt_id,
                        audio_path=str(audio_path),
                        emotion=EMOTION_MAP_4CLASS[emotion_raw],
                        valence=valence,
                        arousal=arousal,
                        dominance=dominance,
                        speaker_id=speaker_id,
                        session_id=session_id,
                        gender=gender,
                        dialog_type=dialog_type,
                    )
                )

    logger.info(
        f"Parsed {len(utterances)} utterances from {session_dir.name}"
    )
    return utterances


def load_iemocap(
    root_dir: str | Path,
    sessions: Optional[List[int]] = None,
) -> List[IEMOCAPUtterance]:
    """Load IEMOCAP utterances across multiple sessions.

    Args:
        root_dir: Root directory containing Session1–Session5 subdirectories.
        sessions: List of session numbers to load (1–5). Defaults to all 5.

    Returns:
        Combined list of IEMOCAPUtterance records from all requested sessions.
    """
    root_dir = Path(root_dir)
    if sessions is None:
        sessions = [1, 2, 3, 4, 5]

    all_utterances: List[IEMOCAPUtterance] = []
    for session_num in sessions:
        session_dir = root_dir / f"Session{session_num}"
        if not session_dir.exists():
            logger.warning(f"Session directory not found, skipping: {session_dir}")
            continue
        utterances = parse_iemocap_session(session_dir)
        all_utterances.extend(utterances)

    logger.info(
        f"Loaded {len(all_utterances)} total IEMOCAP utterances "
        f"from sessions {sessions}"
    )
    return all_utterances


def get_iemocap_cv_splits(
    utterances: List[IEMOCAPUtterance],
) -> List[dict]:
    """Generate 5-fold leave-one-session-out cross-validation splits.

    Args:
        utterances: List of all IEMOCAP utterances.

    Returns:
        List of 5 dicts, each with keys ``train`` and ``test`` containing
        lists of utterance IDs.
    """
    splits = []
    for held_out_session in range(1, 6):
        train_ids = [u.utterance_id for u in utterances if u.session_id != held_out_session]
        test_ids = [u.utterance_id for u in utterances if u.session_id == held_out_session]
        splits.append({"train": train_ids, "test": test_ids})
    return splits
