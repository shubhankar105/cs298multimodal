"""Model checkpoint save/load utilities.

Handles saving and restoring model weights, optimizer state, scheduler
state, and training metadata for resumable training on Apple Silicon.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    best_metric: float = 0.0,
    extra: Optional[dict] = None,
) -> None:
    """Save a training checkpoint to disk.

    The checkpoint is first saved to a temporary file, then atomically
    renamed to avoid corruption from interrupted writes.

    Args:
        path: Destination file path (e.g. ``checkpoints/best.pt``).
        model: The model whose state_dict to save.
        optimizer: Optional optimizer to save state for.
        scheduler: Optional LR scheduler to save state for.
        epoch: Current epoch number.
        best_metric: Best validation metric achieved so far.
        extra: Optional dict of additional metadata to store.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    if extra is not None:
        state["extra"] = extra

    # Atomic write: save to temp then rename
    tmp_path = path.with_suffix(".tmp")
    torch.save(state, tmp_path)
    tmp_path.rename(path)

    logger.info(f"Checkpoint saved: {path} (epoch={epoch}, metric={best_metric:.4f})")


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> dict:
    """Load a training checkpoint from disk.

    Args:
        path: Path to the checkpoint file.
        model: Model to load weights into.
        optimizer: Optional optimizer to restore state for.
        scheduler: Optional scheduler to restore state for.
        device: Device to map tensors to. Defaults to CPU.
        strict: Whether to enforce exact key matching in state_dict.

    Returns:
        Dict with ``epoch``, ``best_metric``, and optional ``extra`` metadata.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    map_location = device if device is not None else torch.device("cpu")
    state = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(state["model_state_dict"], strict=strict)
    logger.info(f"Model weights loaded from {path}")

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
        logger.info("Optimizer state restored")

    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])
        logger.info("Scheduler state restored")

    return {
        "epoch": state.get("epoch", 0),
        "best_metric": state.get("best_metric", 0.0),
        "extra": state.get("extra", {}),
    }


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Optional[Path]:
    """Find the most recently modified checkpoint in a directory.

    Args:
        checkpoint_dir: Directory to search for ``.pt`` files.

    Returns:
        Path to the newest checkpoint, or None if directory is empty.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = sorted(
        checkpoint_dir.glob("*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return checkpoints[0] if checkpoints else None
