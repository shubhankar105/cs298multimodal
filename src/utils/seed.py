"""Reproducibility utilities.

Seeds all random number generators (Python, NumPy, PyTorch) for
deterministic training runs.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Seed all RNGs for reproducibility.

    Sets seeds for: Python stdlib random, NumPy, PyTorch (CPU + MPS/CUDA).
    Also configures PyTorch for deterministic operations where possible.

    Args:
        seed: Integer seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # MPS does not have a separate manual_seed; torch.manual_seed covers it.

    # Enable deterministic algorithms where available
    torch.use_deterministic_algorithms(False)  # Some MPS ops lack deterministic impl
    os.environ["PYTHONHASHSEED"] = str(seed)
