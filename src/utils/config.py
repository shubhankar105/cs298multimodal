"""YAML configuration loader with dataclass mapping.

Loads YAML config files and maps them to structured dataclasses for
type-safe access throughout the MERA pipeline.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class DataConfig:
    max_audio_duration_sec: float = 15.0
    sample_rate: int = 16000
    n_mels: int = 128
    hop_length: int = 160
    win_length: int = 400
    n_fft: int = 512
    f_min: int = 20
    f_max: int = 8000
    num_prosodic_channels: int = 10
    num_hubert_layers: int = 25
    hubert_dim: int = 1024
    pin_memory: bool = False


@dataclass
class PipelineAConfig:
    model_name: str = "microsoft/deberta-v3-base"
    max_text_length: int = 128
    hidden_dim: int = 256
    dropout: float = 0.3
    freeze_layers: int = 6
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    goemotion_epochs: int = 3
    goemotion_batch_size: int = 32
    finetune_epochs: int = 15
    finetune_batch_size: int = 16


@dataclass
class PipelineBConfig:
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 128])
    lstm_hidden: int = 128
    lstm_layers: int = 2
    tcn_channels: int = 64
    tcn_blocks: int = 6
    tcn_kernel_size: int = 5
    prosodic_output_dim: int = 128
    hubert_output_dim: int = 256
    se_reduction: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    batch_size: int = 16


@dataclass
class FusionConfig:
    embed_dim: int = 256
    cross_attention_heads: int = 4
    modality_dropout_prob: float = 0.3
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 20
    batch_size: int = 16


@dataclass
class EndToEndConfig:
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    epochs: int = 5
    batch_size: int = 8
    gradient_accumulation: int = 4
    gradient_checkpointing: bool = True
    unfreeze_deberta_layers: int = 2


@dataclass
class LossConfig:
    lambda_primary: float = 1.0
    lambda_aux_text: float = 0.3
    lambda_aux_audio: float = 0.3
    lambda_consistency: float = 0.2
    label_smoothing: float = 0.1


@dataclass
class SchedulerConfig:
    type: str = "cosine_warmup"
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.01


@dataclass
class EarlyStoppingConfig:
    patience: int = 7
    metric: str = "ua"
    mode: str = "max"


@dataclass
class SpecAugmentConfig:
    freq_mask_param: int = 15
    time_mask_param: int = 50
    n_freq_masks: int = 2
    n_time_masks: int = 2


@dataclass
class AugmentationConfig:
    noise_prob: float = 0.5
    noise_snr_range: List[float] = field(default_factory=lambda: [10.0, 30.0])
    time_stretch_prob: float = 0.3
    time_stretch_range: List[float] = field(default_factory=lambda: [0.9, 1.1])
    pitch_shift_prob: float = 0.3
    pitch_shift_range: List[float] = field(default_factory=lambda: [-2.0, 2.0])
    spec_augment: SpecAugmentConfig = field(default_factory=SpecAugmentConfig)


@dataclass
class LoggingConfig:
    use_wandb: bool = True
    project_name: str = "mera-emotion"
    log_every_n_steps: int = 10
    save_every_n_epochs: int = 1


@dataclass
class MERAConfig:
    """Top-level configuration for the MERA system."""

    seed: int = 42
    device: str = "auto"
    whisper_backend: str = "mlx"
    data: DataConfig = field(default_factory=DataConfig)
    pipeline_a: PipelineAConfig = field(default_factory=PipelineAConfig)
    pipeline_b: PipelineBConfig = field(default_factory=PipelineBConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    end_to_end: EndToEndConfig = field(default_factory=EndToEndConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _dict_to_dataclass(dc_cls: type, data: dict):
    """Recursively convert a dict to a dataclass, ignoring unknown keys."""
    if not isinstance(data, dict):
        return data

    field_names = {f.name for f in dc_cls.__dataclass_fields__.values()}
    filtered = {}
    for key, value in data.items():
        if key not in field_names:
            continue
        field_type = dc_cls.__dataclass_fields__[key].type
        # Resolve string annotations
        if isinstance(field_type, str):
            field_type = eval(field_type)
        # Recurse into nested dataclasses
        if hasattr(field_type, "__dataclass_fields__") and isinstance(value, dict):
            filtered[key] = _dict_to_dataclass(field_type, value)
        else:
            filtered[key] = value
    return dc_cls(**filtered)


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge *overlay* into *base* (overlay wins)."""
    merged = base.copy()
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path, overrides: Optional[dict] = None) -> MERAConfig:
    """Load a YAML config file and return a typed MERAConfig dataclass.

    Supports config inheritance via a ``base_config`` key in the YAML file.
    If ``base_config`` is present, the base config is loaded first and the
    current file's values are merged on top (current wins).

    Args:
        config_path: Path to the YAML configuration file.
        overrides: Optional dict of dot-separated key overrides,
                   e.g. ``{"pipeline_a.learning_rate": 1e-4}``.

    Returns:
        A fully populated MERAConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    # Config inheritance: load base_config first, then merge current on top
    if "base_config" in raw:
        base_path = raw.pop("base_config")
        # Resolve relative to the current config file's directory
        base_path = (config_path.parent / base_path).resolve()
        base_config = load_config(base_path)
        import dataclasses
        base_raw = dataclasses.asdict(base_config)
        raw = _deep_merge(base_raw, raw)

    # Apply dot-separated overrides
    if overrides:
        for key, value in overrides.items():
            parts = key.split(".")
            d = raw
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = value

    return _dict_to_dataclass(MERAConfig, raw)


def save_config(config: MERAConfig, save_path: str | Path) -> None:
    """Save a MERAConfig to a YAML file.

    Args:
        config: The configuration to serialize.
        save_path: Destination file path.
    """
    import dataclasses

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    raw = dataclasses.asdict(config)
    with open(save_path, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)
