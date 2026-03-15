"""Logging and Weights & Biases integration.

Provides a configured Python logger and optional W&B experiment tracking.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional


def get_logger(name: str, log_file: Optional[str | Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Create a configured logger with console and optional file output.

    Args:
        name: Logger name (typically ``__name__``).
        log_file: Optional path for file logging output.
        level: Logging level. Defaults to INFO.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class WandbLogger:
    """Thin wrapper around Weights & Biases for experiment tracking.

    Handles graceful fallback when W&B is not installed or disabled.
    """

    def __init__(
        self,
        project_name: str = "mera-emotion",
        run_name: Optional[str] = None,
        config: Optional[dict] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self._run = None

        if not enabled:
            return

        try:
            import wandb

            self._wandb = wandb
            self._run = wandb.init(
                project=project_name,
                name=run_name,
                config=config,
                reinit=True,
            )
        except ImportError:
            logging.getLogger(__name__).warning(
                "wandb not installed. Disabling W&B logging."
            )
            self.enabled = False
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Failed to initialize W&B: {e}. Disabling W&B logging."
            )
            self.enabled = False

    def log(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        """Log a dict of metrics to W&B.

        Args:
            metrics: Key-value pairs to log.
            step: Optional global step number.
        """
        if self.enabled and self._run is not None:
            self._wandb.log(metrics, step=step)

    def log_summary(self, metrics: dict[str, Any]) -> None:
        """Log summary metrics (reported once at end of run).

        Args:
            metrics: Key-value pairs for the run summary.
        """
        if self.enabled and self._run is not None:
            for key, value in metrics.items():
                self._run.summary[key] = value

    def finish(self) -> None:
        """Finalize the W&B run."""
        if self.enabled and self._run is not None:
            self._run.finish()
            self._run = None
