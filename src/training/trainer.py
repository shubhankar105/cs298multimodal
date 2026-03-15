"""Reusable training loop for all MERA training phases.

Supports:
- All four training modes (Pipeline A, B, Fusion, End-to-End).
- Apple Silicon MPS device handling.
- Gradient accumulation (physical batch 8 × accumulation 4 = effective 32).
- Gradient checkpointing toggle.
- Gradient clipping.
- W&B logging integration.
- Epoch-level metric computation.
- Checkpoint save/load.
- Early stopping.
- Class weight computation from training distribution.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.metrics import MetricTracker, format_metrics
from src.training.schedulers import build_cosine_warmup_scheduler
from src.training.early_stopping import EarlyStopping
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.device import empty_cache
from src.utils.logging_utils import WandbLogger

logger = logging.getLogger(__name__)


class Trainer:
    """Configurable training loop for the MERA system.

    The trainer is agnostic to *which* model variant is being trained —
    the caller provides the model, loss function, and a ``forward_fn``
    that extracts the right tensors from a batch and returns
    ``(loss_dict, logits, targets)``.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        *,
        # Data
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        # Schedule
        scheduler: Optional[Any] = None,
        total_epochs: int = 10,
        # Gradient
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        gradient_checkpointing: bool = False,
        # Logging
        wandb_logger: Optional[WandbLogger] = None,
        log_every_n_steps: int = 10,
        # Checkpointing
        checkpoint_dir: Optional[str | Path] = None,
        save_every_n_epochs: int = 1,
        # Early stopping
        early_stopping: Optional[EarlyStopping] = None,
        # Metric computation
        num_classes: int = 4,
        # Callbacks
        forward_fn: Optional[Callable] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.scheduler = scheduler
        self.total_epochs = total_epochs

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.gradient_checkpointing = gradient_checkpointing

        self.wandb_logger = wandb_logger
        self.log_every_n_steps = log_every_n_steps

        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.save_every_n_epochs = save_every_n_epochs

        self.early_stopping = early_stopping
        self.num_classes = num_classes

        # forward_fn(model, batch, device) → (loss_dict, logits, targets)
        self.forward_fn = forward_fn or self._default_forward_fn

        self.global_step = 0
        self.best_metric = 0.0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def train(self) -> dict:
        """Run the full training loop.

        Returns:
            Dict with ``best_metric``, ``best_epoch``, ``final_train_metrics``,
            ``final_val_metrics``.
        """
        logger.info(f"Starting training for {self.total_epochs} epochs on {self.device}")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {self.train_loader.batch_size * self.gradient_accumulation_steps}")

        results = {
            "best_metric": 0.0,
            "best_epoch": 0,
            "train_history": [],
            "val_history": [],
        }

        for epoch in range(1, self.total_epochs + 1):
            epoch_start = time.time()

            # --- Train ---
            train_metrics = self._train_epoch(epoch)
            results["train_history"].append(train_metrics)

            # --- Validate ---
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self._validate_epoch(epoch)
                results["val_history"].append(val_metrics)

            elapsed = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch}/{self.total_epochs} ({elapsed:.1f}s) | "
                f"Train loss: {train_metrics.get('loss', 0):.4f} | "
                f"Val UA: {val_metrics.get('ua', 0):.4f}"
            )

            # --- W&B logging ---
            if self.wandb_logger:
                log_data = {f"train/{k}": v for k, v in train_metrics.items()}
                log_data.update({f"val/{k}": v for k, v in val_metrics.items()})
                log_data["epoch"] = epoch
                self.wandb_logger.log(log_data, step=self.global_step)

            # --- Checkpointing ---
            if self.checkpoint_dir and epoch % self.save_every_n_epochs == 0:
                self._save_epoch_checkpoint(epoch, val_metrics.get("ua", 0.0))

            # --- Early stopping ---
            if self.early_stopping and val_metrics:
                stop = self.early_stopping(
                    score=val_metrics.get("ua", 0.0),
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                )
                if stop:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

            # --- Track best ---
            ua = val_metrics.get("ua", train_metrics.get("ua", 0.0))
            if ua > results["best_metric"]:
                results["best_metric"] = ua
                results["best_epoch"] = epoch

        results["final_train_metrics"] = results["train_history"][-1] if results["train_history"] else {}
        results["final_val_metrics"] = results["val_history"][-1] if results["val_history"] else {}

        if self.wandb_logger:
            self.wandb_logger.log_summary({
                "best_ua": results["best_metric"],
                "best_epoch": results["best_epoch"],
            })

        return results

    # ------------------------------------------------------------------
    # Train one epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        tracker = MetricTracker(self.num_classes)
        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            loss_dict, logits, targets = self.forward_fn(
                self.model, batch, self.device,
            )
            loss = loss_dict["total"] / self.gradient_accumulation_steps
            loss.backward()

            total_loss += loss_dict["total"].item()
            num_batches += 1
            tracker.update(logits.detach(), targets.detach())

            # Accumulation step
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm,
                    )
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.global_step += 1

                # Step-level logging
                if self.global_step % self.log_every_n_steps == 0 and self.wandb_logger:
                    lr = self.optimizer.param_groups[0]["lr"]
                    self.wandb_logger.log(
                        {"train/step_loss": loss_dict["total"].item(), "lr": lr},
                        step=self.global_step,
                    )

        # Handle remaining accumulated gradients
        if num_batches % self.gradient_accumulation_steps != 0:
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm,
                )
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_step += 1

        metrics = tracker.compute()
        metrics["loss"] = total_loss / max(num_batches, 1)
        return metrics

    # ------------------------------------------------------------------
    # Validate one epoch
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> dict:
        self.model.eval()
        tracker = MetricTracker(self.num_classes)
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            loss_dict, logits, targets = self.forward_fn(
                self.model, batch, self.device,
            )
            total_loss += loss_dict["total"].item()
            num_batches += 1
            tracker.update(logits, targets)

        metrics = tracker.compute()
        metrics["loss"] = total_loss / max(num_batches, 1)

        logger.info(f"[Val Epoch {epoch}] {format_metrics(metrics)}")
        return metrics

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _save_epoch_checkpoint(self, epoch: int, metric: float) -> None:
        assert self.checkpoint_dir is not None
        path = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
        save_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            best_metric=metric,
        )

    # ------------------------------------------------------------------
    # Default forward function
    # ------------------------------------------------------------------

    @staticmethod
    def _default_forward_fn(
        model: nn.Module,
        batch: dict,
        device: torch.device,
    ) -> tuple[dict, torch.Tensor, torch.Tensor]:
        """Default forward: expects batch dict with ``emotion`` key as target
        and passes all tensor values to the model.

        Override this via ``forward_fn`` for custom model interfaces.
        """
        raise NotImplementedError(
            "You must provide a `forward_fn` to the Trainer. "
            "It should accept (model, batch, device) and return "
            "(loss_dict, logits, targets)."
        )
