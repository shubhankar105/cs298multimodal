"""Learning rate schedulers for MERA training.

Implements a **linear warmup + cosine annealing** schedule:

1. **Warmup** (first ``warmup_ratio`` of total steps):
   LR ramps linearly from 0 to ``peak_lr``.
2. **Cosine decay** (remaining steps):
   LR decays from ``peak_lr`` to ``peak_lr × min_lr_ratio`` following
   a half-cosine curve.

This is the standard schedule for fine-tuning transformers and is used
across all MERA training phases.
"""

from __future__ import annotations

import math
from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_cosine_warmup_scheduler(
    optimizer: Optimizer,
    total_steps: int,
    warmup_ratio: float = 0.1,
    min_lr_ratio: float = 0.01,
    last_epoch: int = -1,
) -> LambdaLR:
    """Build a linear-warmup + cosine-decay LR scheduler.

    Args:
        optimizer: The optimizer whose LR will be adjusted.
        total_steps: Total number of training steps (across all epochs).
        warmup_ratio: Fraction of ``total_steps`` used for warmup.
        min_lr_ratio: Minimum LR as a fraction of peak LR.
        last_epoch: The index of last epoch (for resuming). -1 = fresh start.

    Returns:
        A ``torch.optim.lr_scheduler.LambdaLR`` instance.
    """
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup: 0 → 1
            return current_step / max(1, warmup_steps)
        else:
            # Cosine decay: 1 → min_lr_ratio
            progress = (current_step - warmup_steps) / max(
                1, total_steps - warmup_steps
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def build_scheduler_from_config(
    optimizer: Optimizer,
    total_steps: int,
    config,
    last_epoch: int = -1,
) -> LambdaLR:
    """Build a scheduler from a ``SchedulerConfig`` dataclass.

    Args:
        optimizer: Optimizer.
        total_steps: Total training steps.
        config: ``SchedulerConfig`` (from ``src.utils.config``).
        last_epoch: For resuming.

    Returns:
        Configured LambdaLR scheduler.
    """
    if config.type == "cosine_warmup":
        return build_cosine_warmup_scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_ratio=config.warmup_ratio,
            min_lr_ratio=config.min_lr_ratio,
            last_epoch=last_epoch,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {config.type}")
