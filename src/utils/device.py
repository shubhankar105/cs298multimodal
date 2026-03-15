"""Cross-platform device detection and GPU configuration.

Supports CUDA (Linux / Colab), MPS (Apple Silicon), and CPU fallback.
Provides utilities for configuring memory management on each backend.
"""

from __future__ import annotations

import os
import platform

import torch


# MPS memory watermark settings for unified memory management
MPS_SETTINGS = {
    "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.7",
    "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.5",
}


def configure_mps_memory() -> None:
    """Set MPS memory watermark environment variables.

    Only applies when running on an MPS-capable system.
    Should be called before any PyTorch operations to ensure
    the MPS backend respects memory limits on unified memory.
    On CUDA / CPU this is a no-op.
    """
    if not torch.backends.mps.is_available():
        return
    for key, value in MPS_SETTINGS.items():
        os.environ.setdefault(key, value)


def get_device(device_override: str = "auto") -> torch.device:
    """Get the best available accelerator device.

    Priority order: CUDA > MPS > CPU.

    Args:
        device_override: Explicit device string (``"cuda"``, ``"mps"``,
            ``"cpu"``).  Use ``"auto"`` (default) for automatic detection.

    Returns:
        torch.device configured for the current hardware.
    """
    if device_override != "auto":
        return torch.device(device_override)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_optimal_dtype() -> torch.dtype:
    """Get the optimal dtype for training on the current device.

    float32 is more stable on MPS than float16 for training.
    Use float16 only for inference or frozen model components.

    Returns:
        torch.float32 for training stability on MPS.
    """
    return torch.float32


def get_device_info() -> dict:
    """Gather device and platform information for logging.

    Returns:
        Dict with keys: device, platform, processor, python_version,
        torch_version, mps_available, cuda_available.
    """
    return {
        "device": str(get_device()),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }


def empty_cache() -> None:
    """Clear MPS or CUDA cache to free GPU memory."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
