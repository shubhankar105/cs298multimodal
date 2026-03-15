#!/usr/bin/env python3
"""Train Pipeline B (Audio emotion) standalone on cached features.

Uses pre-computed spectrograms, prosodic contours, and HuBERT embeddings.
30 epochs, 5-fold leave-one-session-out cross-validation.

Usage::

    python scripts/train_pipeline_b.py --config configs/default.yaml
    python scripts/train_pipeline_b.py --config configs/default.yaml --fold 0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.device import configure_mps_memory, get_device
from src.utils.seed import seed_everything
from src.utils.logging_utils import get_logger
from src.data.collate import CachedFeatureDataset, collate_features
from src.data.augmentation import build_spec_augment_from_config
from src.training.losses import PipelineLoss, compute_class_weights
from src.training.schedulers import build_cosine_warmup_scheduler
from src.training.early_stopping import EarlyStopping
from src.training.trainer import Trainer

logger = get_logger(__name__)


def pipeline_b_forward(model, batch, device):
    """Forward function for Pipeline B training."""
    spec = batch["spectrogram"].to(device)
    spec_mask = batch["spectrogram_mask"].to(device)
    pros = batch["prosody"].to(device)
    pros_mask = batch["prosody_mask"].to(device)
    hub = batch["hubert"].to(device)
    hub_mask = batch["hubert_mask"].to(device)
    targets = batch["emotion"].to(device)

    logits, _ = model(spec, spec_mask, pros, pros_mask, hub, hub_mask)

    loss_fn = pipeline_b_forward._loss_fn
    loss_dict = loss_fn(logits, targets)
    return loss_dict, logits, targets


def load_iemocap_metadata(project_root: Path) -> pd.DataFrame:
    """Load metadata and add session column."""
    metadata_path = project_root / "data" / "processed" / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}. Run preprocess_all.py first.")

    metadata = pd.read_csv(metadata_path)
    iemocap = metadata[metadata["dataset"] == "iemocap"].reset_index(drop=True)

    sessions = []
    for fid in iemocap["file_id"]:
        try:
            sess = int(fid.split("Ses")[1][:2])
        except (IndexError, ValueError):
            sess = 1
        sessions.append(sess)
    iemocap["session"] = sessions
    return iemocap


def main():
    parser = argparse.ArgumentParser(description="Train Pipeline B (Audio)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--fold", type=int, default=None)
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    configure_mps_memory()
    seed_everything(config.seed)
    device = get_device(config.device)

    logger.info(f"Device: {device}")

    iemocap = load_iemocap_metadata(PROJECT_ROOT)
    if iemocap.empty:
        logger.error("No IEMOCAP entries found.")
        sys.exit(1)

    folds = range(1, 6) if args.fold is None else [args.fold + 1]

    for fold_session in folds:
        logger.info(f"\n--- Fold {fold_session} (Test=Session {fold_session}) ---")
        seed_everything(config.seed)

        train_df = iemocap[iemocap["session"] != fold_session].reset_index(drop=True)
        val_df = iemocap[iemocap["session"] == fold_session].reset_index(drop=True)

        spec_aug = build_spec_augment_from_config(config.augmentation)
        train_ds = CachedFeatureDataset(train_df, spec_augment=spec_aug)
        val_ds = CachedFeatureDataset(val_df)

        train_loader = DataLoader(
            train_ds, batch_size=config.pipeline_b.batch_size,
            shuffle=True, num_workers=0, collate_fn=collate_features,
            pin_memory=config.data.pin_memory,
        )
        val_loader = DataLoader(
            val_ds, batch_size=config.pipeline_b.batch_size,
            shuffle=False, num_workers=0, collate_fn=collate_features,
            pin_memory=config.data.pin_memory,
        )

        # Build model
        from src.models.pipeline_b.audio_emotion_head import AudioEmotionHead
        model = AudioEmotionHead(
            num_classes=4, embed_dim=config.fusion.embed_dim,
            dropout=config.pipeline_a.dropout,
            tcn_channels=config.pipeline_b.tcn_channels,
            tcn_blocks=config.pipeline_b.tcn_blocks,
            tcn_kernel_size=config.pipeline_b.tcn_kernel_size,
            prosodic_output_dim=config.pipeline_b.prosodic_output_dim,
            se_reduction=config.pipeline_b.se_reduction,
        )
        model.float()
        model.to(device)

        class_weights = compute_class_weights(train_ds.metadata["emotion_4class"].map(
            {"angry": 0, "happy": 1, "sad": 2, "neutral": 3}
        ).tolist())
        loss_fn = PipelineLoss(class_weights=class_weights.to(device), label_smoothing=config.loss.label_smoothing)
        pipeline_b_forward._loss_fn = loss_fn

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.pipeline_b.learning_rate,
            weight_decay=config.pipeline_b.weight_decay,
        )
        total_steps = len(train_loader) * config.pipeline_b.epochs
        scheduler = build_cosine_warmup_scheduler(
            optimizer, total_steps,
            warmup_ratio=config.scheduler.warmup_ratio,
            min_lr_ratio=config.scheduler.min_lr_ratio,
        )

        ckpt_dir = PROJECT_ROOT / "checkpoints" / "pipeline_b" / f"fold_{fold_session}"
        es = EarlyStopping(
            patience=config.early_stopping.patience, metric="ua", mode="max",
            checkpoint_path=ckpt_dir / "best.pt",
        )

        trainer = Trainer(
            model=model, optimizer=optimizer, loss_fn=loss_fn, device=device,
            train_loader=train_loader, val_loader=val_loader,
            scheduler=scheduler, total_epochs=config.pipeline_b.epochs,
            early_stopping=es, forward_fn=pipeline_b_forward, num_classes=4,
            checkpoint_dir=ckpt_dir,
        )
        results = trainer.train()
        logger.info(f"Fold {fold_session} best UA: {results['best_metric']:.4f}")

    logger.info("Pipeline B training complete.")


if __name__ == "__main__":
    main()
