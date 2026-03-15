"""Whisper transcription with dual backend support.

Backends:
- **mlx** (default on Apple Silicon): Uses ``mlx-community/whisper-large-v3-mlx``
  via the ``mlx-whisper`` package for optimised Metal inference.
- **huggingface** (CUDA / CPU): Uses ``openai/whisper-large-v3`` via the
  ``transformers`` pipeline with automatic device placement.

The backend is selected via the ``backend`` parameter (``"mlx"`` or
``"huggingface"``).  When not specified, ``"mlx"`` is used.

Output per utterance: a JSON file containing
- ``text``:       full transcript string
- ``segments``:   list of ``{start, end, text, words}`` with word-level timestamps
- ``language``:   detected language code
- ``file_id``:    stem of the source audio file

The transcriber supports **skip-if-exists** logic so interrupted runs can be
resumed without reprocessing already-finished files.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

MLX_MODEL_ID = "mlx-community/whisper-large-v3-mlx"
HF_MODEL_ID = "openai/whisper-large-v3"


def _transcribe_mlx(
    audio_path: str,
    model_id: str,
    language: str,
    fp16: bool,
) -> dict:
    """Transcribe using the mlx-whisper backend (Apple Silicon)."""
    import mlx_whisper  # Lazy import — heavy dependency

    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=model_id,
        language=language,
        word_timestamps=True,
        fp16=fp16,
    )
    return result


def _transcribe_huggingface(
    audio_path: str,
    model_id: str,
    language: str,
) -> dict:
    """Transcribe using the HuggingFace transformers pipeline (CUDA / CPU)."""
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    result = pipe(
        audio_path,
        generate_kwargs={"language": language},
        return_timestamps="word",
    )

    # Normalise to the same dict shape that mlx_whisper returns
    segments = []
    if "chunks" in result:
        for chunk in result["chunks"]:
            ts = chunk.get("timestamp", (None, None))
            segments.append({
                "start": ts[0] if ts[0] is not None else 0.0,
                "end": ts[1] if ts[1] is not None else 0.0,
                "text": chunk.get("text", ""),
                "words": [],
            })

    return {
        "text": result.get("text", ""),
        "segments": segments,
        "language": language,
    }


def transcribe_file(
    audio_path: str | Path,
    output_dir: str | Path,
    model_id: str | None = None,
    language: str = "en",
    fp16: bool = True,
    backend: str = "mlx",
) -> Optional[dict]:
    """Transcribe a single audio file and write the result to JSON.

    Args:
        audio_path: Path to the input audio (WAV / FLAC / MP3 ...).
        output_dir: Directory where the JSON will be written.
        model_id: HuggingFace / MLX model identifier.  If ``None``, a
            sensible default for the chosen backend is used.
        language: ISO language code for Whisper.
        fp16: Use float16 inference (relevant for mlx backend).
        backend: ``"mlx"`` (Apple Silicon) or ``"huggingface"`` (CUDA / CPU).

    Returns:
        The transcript dict if successful, *None* if the file already exists.
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_id = audio_path.stem
    output_path = output_dir / f"{file_id}.json"

    if output_path.exists():
        logger.debug(f"Skipping (already exists): {file_id}")
        return None

    # Select default model id if not provided
    if model_id is None:
        model_id = MLX_MODEL_ID if backend == "mlx" else HF_MODEL_ID

    start = time.time()

    if backend == "huggingface":
        result = _transcribe_huggingface(str(audio_path), model_id, language)
    else:
        result = _transcribe_mlx(str(audio_path), model_id, language, fp16)

    elapsed = time.time() - start

    output = {
        "file_id": file_id,
        "text": result["text"].strip(),
        "segments": [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "words": seg.get("words", []),
            }
            for seg in result.get("segments", [])
        ],
        "language": result.get("language", language),
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.debug(f"Transcribed {file_id} in {elapsed:.1f}s ({len(output['text'])} chars)")
    return output


def transcribe_batch(
    audio_paths: List[str | Path],
    output_dir: str | Path,
    model_id: str | None = None,
    language: str = "en",
    fp16: bool = True,
    show_progress: bool = True,
    backend: str = "mlx",
) -> dict:
    """Transcribe a list of audio files, saving one JSON per utterance.

    Files whose JSON output already exists on disk are silently skipped,
    enabling resumable processing.

    Args:
        audio_paths: Iterable of paths to audio files.
        output_dir: Directory to write output JSONs into.
        model_id: HuggingFace / MLX model identifier.
        language: ISO language code.
        fp16: Use float16 for inference.
        show_progress: Show a tqdm progress bar.
        backend: ``"mlx"`` or ``"huggingface"``.

    Returns:
        Dict with keys ``processed`` (int), ``skipped`` (int), ``errors`` (list of str).
    """
    from tqdm import tqdm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"processed": 0, "skipped": 0, "errors": []}
    iterator = tqdm(audio_paths, desc="Transcribing", disable=not show_progress)

    for audio_path in iterator:
        audio_path = Path(audio_path)
        file_id = audio_path.stem
        output_path = output_dir / f"{file_id}.json"

        if output_path.exists():
            stats["skipped"] += 1
            continue

        try:
            transcribe_file(
                audio_path,
                output_dir,
                model_id=model_id,
                language=language,
                fp16=fp16,
                backend=backend,
            )
            stats["processed"] += 1
        except Exception as e:
            logger.error(f"Failed to transcribe {file_id}: {e}")
            stats["errors"].append(str(audio_path))

    logger.info(
        f"Transcription complete: {stats['processed']} processed, "
        f"{stats['skipped']} skipped, {len(stats['errors'])} errors"
    )
    return stats


def load_transcript(transcript_path: str | Path) -> dict:
    """Load a previously saved transcript JSON.

    Args:
        transcript_path: Path to the JSON file.

    Returns:
        Dict with keys ``file_id``, ``text``, ``segments``, ``language``.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    transcript_path = Path(transcript_path)
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")

    with open(transcript_path, "r") as f:
        return json.load(f)
