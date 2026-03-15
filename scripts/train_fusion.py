#!/usr/bin/env python3
"""Train the fusion module (cross-modal attention + gated fusion + classifier).

Loads pre-trained Pipeline A and Pipeline B checkpoints, freezes both,
and trains only the fusion components for 20 epochs with 5-fold CV.

Usage::

    python scripts/train_fusion.py --config configs/default.yaml
    python scripts/train_fusion.py --config configs/default.yaml --fold 0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.device import configure_mps_memory, get_device
from src.utils.seed import seed_everything
from src.utils.logging_utils import get_logger
from src.utils.checkpoint import load_checkpoint
from src.data.collate import CachedFeatureDataset, collate_features, EMOTION_TO_IDX
from src.training.losses import MERALoss, compute_class_weights
from src.training.schedulers import build_cosine_warmup_scheduler
from src.training.early_stopping import EarlyStopping
from src.training.trainer import Trainer
from src.models.fusion.mera_model import MERAModel, TrainingMode

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Multimodal dataset: extends CachedFeatureDataset with tokenised text
# ---------------------------------------------------------------------------

class MultimodalDataset(Dataset):
    """Wraps cached audio features and adds tokenised transcript text."""

    def __init__(self, metadata_df, tokenizer, max_length=128, spec_augment=None):
        self.audio_ds = CachedFeatureDataset(metadata_df, spec_augment=spec_augment)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.metadata = metadata_df.reset_index(drop=True)

        # Pre-load all transcripts
        self.texts = []
        for _, row in self.metadata.iterrows():
            tpath = Path(row["transcript_path"])
            if tpath.exists():
                with open(tpath) as f:
                    self.texts.append(json.load(f).get("text", ""))
            else:
                self.texts.append("")

    def __len__(self):
        return len(self.audio_ds)

    def __getitem__(self, idx):
        sample = self.audio_ds[idx]
        sample["raw_text"] = self.texts[idx]
        return sample


def multimodal_collate(batch, tokenizer, max_length=128, device=None):
    """Collate audio features + tokenise text."""
    audio_batch = collate_features(batch)

    texts = [b["raw_text"] for b in batch]
    encoded = tokenizer(
        texts, padding=True, truncation=True,
        max_length=max_length, return_tensors="pt",
    )
    audio_batch["input_ids"] = encoded["input_ids"]
    audio_batch["attention_mask"] = encoded["attention_mask"]
    return audio_batch


def fusion_forward(model, batch, device):
    """Forward function for full MERA model."""
    outputs = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        spectrogram=batch["spectrogram"].to(device),
        spec_mask=batch["spectrogram_mask"].to(device),
        prosody=batch["prosody"].to(device),
        pros_mask=batch["prosody_mask"].to(device),
        hubert=batch["hubert"].to(device),
        hub_mask=batch["hubert_mask"].to(device),
    )
    targets = batch["emotion"].to(device)

    loss_fn = fusion_forward._loss_fn
    loss_dict = loss_fn(outputs, targets)
    return loss_dict, outputs["final_logits"], targets


def main():
    parser = argparse.ArgumentParser(description="Train Fusion Module")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--pipeline-a-ckpt", type=str, default=None, help="Path to Pipeline A checkpoint")
    parser.add_argument("--pipeline-b-ckpt", type=str, default=None, help="Path to Pipeline B checkpoint")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    configure_mps_memory()
    seed_everything(config.seed)
    device = get_device(config.device)
    logger.info(f"Device: {device}")

    # Load metadata
    metadata_path = PROJECT_ROOT / "data" / "processed" / "metadata.csv"
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

    folds = range(1, 6) if args.fold is None else [args.fold + 1]

    for fold_session in folds:
        logger.info(f"\n--- Fold {fold_session} (Test=Session {fold_session}) ---")
        seed_everything(config.seed)

        train_df = iemocap[iemocap["session"] != fold_session].reset_index(drop=True)
        val_df = iemocap[iemocap["session"] == fold_session].reset_index(drop=True)

        # Build MERA model
        model = MERAModel(
            num_classes=4,
            text_model_name=config.pipeline_a.model_name,
            embed_dim=config.fusion.embed_dim,
            dropout=config.pipeline_a.dropout,
            modality_dropout_prob=config.fusion.modality_dropout_prob,
            tcn_channels=config.pipeline_b.tcn_channels,
            tcn_blocks=config.pipeline_b.tcn_blocks,
            tcn_kernel_size=config.pipeline_b.tcn_kernel_size,
            prosodic_output_dim=config.pipeline_b.prosodic_output_dim,
            se_reduction=config.pipeline_b.se_reduction,
            cross_attention_heads=config.fusion.cross_attention_heads,
        )
        model.float()
        model.to(device)

        # Load pre-trained checkpoints if available
        pa_ckpt = args.pipeline_a_ckpt or str(
            PROJECT_ROOT / "checkpoints" / "pipeline_a" / f"fold_{fold_session}" / "best.pt"
        )
        pb_ckpt = args.pipeline_b_ckpt or str(
            PROJECT_ROOT / "checkpoints" / "pipeline_b" / f"fold_{fold_session}" / "best.pt"
        )

        if Path(pa_ckpt).exists():
            load_checkpoint(pa_ckpt, model.pipeline_a, strict=False)
            logger.info(f"Loaded Pipeline A from {pa_ckpt}")
        else:
            logger.warning(f"Pipeline A checkpoint not found: {pa_ckpt}")

        if Path(pb_ckpt).exists():
            load_checkpoint(pb_ckpt, model.pipeline_b, strict=False)
            logger.info(f"Loaded Pipeline B from {pb_ckpt}")
        else:
            logger.warning(f"Pipeline B checkpoint not found: {pb_ckpt}")

        # Freeze pipelines, train fusion only
        model.set_training_mode(TrainingMode.FUSION_ONLY)

        # Datasets
        tokenizer = model.pipeline_a._tokenizer
        train_ds = MultimodalDataset(train_df, tokenizer)
        val_ds = MultimodalDataset(val_df, tokenizer)

        from functools import partial
        collate = partial(multimodal_collate, tokenizer=tokenizer, max_length=config.pipeline_a.max_text_length)

        train_loader = DataLoader(train_ds, batch_size=config.fusion.batch_size, shuffle=True, num_workers=0, collate_fn=collate, pin_memory=config.data.pin_memory)
        val_loader = DataLoader(val_ds, batch_size=config.fusion.batch_size, shuffle=False, num_workers=0, collate_fn=collate, pin_memory=config.data.pin_memory)

        # Loss
        train_labels = train_df["emotion_4class"].map(EMOTION_TO_IDX).tolist()
        class_weights = compute_class_weights(train_labels).to(device)
        loss_fn = MERALoss(class_weights=class_weights, label_smoothing=config.loss.label_smoothing,
                           lambda_primary=config.loss.lambda_primary,
                           lambda_aux_text=config.loss.lambda_aux_text,
                           lambda_aux_audio=config.loss.lambda_aux_audio,
                           lambda_consistency=config.loss.lambda_consistency)
        fusion_forward._loss_fn = loss_fn

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=config.fusion.learning_rate, weight_decay=config.fusion.weight_decay)

        total_steps = len(train_loader) * config.fusion.epochs
        scheduler = build_cosine_warmup_scheduler(optimizer, total_steps)

        ckpt_dir = PROJECT_ROOT / "checkpoints" / "fusion" / f"fold_{fold_session}"
        es = EarlyStopping(patience=config.early_stopping.patience, metric="ua", mode="max", checkpoint_path=ckpt_dir / "best.pt")

        trainer = Trainer(
            model=model, optimizer=optimizer, loss_fn=loss_fn, device=device,
            train_loader=train_loader, val_loader=val_loader,
            scheduler=scheduler, total_epochs=config.fusion.epochs,
            early_stopping=es, forward_fn=fusion_forward, num_classes=4,
            checkpoint_dir=ckpt_dir,
        )
        results = trainer.train()
        logger.info(f"Fold {fold_session} best UA: {results['best_metric']:.4f}")

    logger.info("Fusion training complete.")


if __name__ == "__main__":
    main()
