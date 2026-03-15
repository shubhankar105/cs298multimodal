"""Patience-based early stopping.

Monitors a validation metric (default: unweighted accuracy / UA) and
stops training if no improvement is seen for ``patience`` consecutive
epochs.  Optionally saves the best checkpoint automatically.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch.nn as nn

from src.utils.checkpoint import save_checkpoint

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping with best-model checkpointing.

    Args:
        patience: Number of epochs without improvement before stopping.
        metric: Name of the metric being tracked (for logging).
        mode: ``"max"`` if higher is better, ``"min"`` if lower is better.
        min_delta: Minimum change to qualify as an improvement.
        checkpoint_path: If provided, save the best model here.
    """

    def __init__(
        self,
        patience: int = 7,
        metric: str = "ua",
        mode: str = "max",
        min_delta: float = 0.0,
        checkpoint_path: Optional[str | Path] = None,
    ):
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

        self.best_score: Optional[float] = None
        self.best_epoch: int = 0
        self.counter: int = 0
        self.should_stop: bool = False

    def __call__(
        self,
        score: float,
        epoch: int,
        model: Optional[nn.Module] = None,
        optimizer=None,
        scheduler=None,
        extra: Optional[dict] = None,
    ) -> bool:
        """Check whether training should stop.

        Args:
            score: Current epoch's metric value.
            epoch: Current epoch number.
            model: If provided and score is best, save checkpoint.
            optimizer: Optionally save optimizer state.
            scheduler: Optionally save scheduler state.
            extra: Extra metadata to include in checkpoint.

        Returns:
            ``True`` if training should stop, ``False`` otherwise.
        """
        improved = self._is_improvement(score)

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0

            # Save best checkpoint
            if model is not None and self.checkpoint_path is not None:
                save_checkpoint(
                    path=self.checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_metric=score,
                    extra=extra,
                )
                logger.info(
                    f"New best {self.metric} = {score:.4f} at epoch {epoch}. "
                    f"Checkpoint saved to {self.checkpoint_path}"
                )
        else:
            self.counter += 1
            logger.info(
                f"No improvement in {self.metric} for {self.counter}/{self.patience} epochs "
                f"(best = {self.best_score:.4f} at epoch {self.best_epoch})"
            )

        self.should_stop = self.counter >= self.patience
        return self.should_stop

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta

    def state_dict(self) -> dict:
        """Serialise state for checkpoint resuming."""
        return {
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "counter": self.counter,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore state from a checkpoint."""
        self.best_score = state["best_score"]
        self.best_epoch = state["best_epoch"]
        self.counter = state["counter"]
        self.should_stop = state["should_stop"]
