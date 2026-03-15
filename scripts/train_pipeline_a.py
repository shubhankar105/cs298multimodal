#!/usr/bin/env python3
"""Train Pipeline A (DeBERTa text emotion classifier) standalone.

Training plan:
1. Pre-train on GoEmotions (mapped to 4 classes) for 3 epochs.
2. Fine-tune on IEMOCAP transcripts for 15 epochs with 5-fold
   leave-one-session-out cross-validation.

Usage::

    python scripts/train_pipeline_a.py --config configs/default.yaml
    python scripts/train_pipeline_a.py --config configs/default.yaml --skip-goemotion
"""

from __future__ import annotations

import argparse
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
from src.utils.logging_utils import get_logger, WandbLogger
from src.training.losses import PipelineLoss, compute_class_weights
from src.training.metrics import MetricTracker
from src.training.schedulers import build_cosine_warmup_scheduler
from src.training.early_stopping import EarlyStopping
from src.training.trainer import Trainer

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Text-only dataset
# ---------------------------------------------------------------------------

class TextEmotionDataset(Dataset):
    """Simple dataset wrapping tokenised text + emotion labels."""

    EMOTION_TO_IDX = {"angry": 0, "happy": 1, "sad": 2, "neutral": 3}

    def __init__(self, texts: list[str], labels: list[str], tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        self.labels = torch.tensor([self.EMOTION_TO_IDX[l] for l in labels], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "emotion": self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Forward function for Trainer
# ---------------------------------------------------------------------------

def pipeline_a_forward(model, batch, device):
    """Forward function: run text encoder, compute loss."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    targets = batch["emotion"].to(device)

    logits, _, _ = model(input_ids, attention_mask)

    # Loss is computed outside via trainer's loss_fn
    loss_fn = pipeline_a_forward._loss_fn
    loss_dict = loss_fn(logits, targets)

    return loss_dict, logits, targets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Pipeline A (Text)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--skip-goemotion", action="store_true", help="Skip GoEmotions pre-training")
    parser.add_argument("--fold", type=int, default=None, help="Run single fold (0-4), or all if omitted")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    configure_mps_memory()
    seed_everything(config.seed)
    device = get_device(config.device)

    logger.info(f"Device: {device}")
    logger.info(f"Config: {args.config}")

    # ---- Build model ----
    from src.models.pipeline_a.text_encoder import TextEmotionEncoder

    model = TextEmotionEncoder(
        model_name=config.pipeline_a.model_name,
        num_classes=4,
        hidden_dim=config.pipeline_a.hidden_dim,
        dropout=config.pipeline_a.dropout,
        freeze_layers=config.pipeline_a.freeze_layers,
    )
    model.float()
    model.to(device)

    # ---- Step 1: GoEmotions pre-training ----
    if not args.skip_goemotion:
        logger.info("=" * 60)
        logger.info("Step 1: GoEmotions Pre-training")
        logger.info("=" * 60)

        try:
            from src.data.goemotions_loader import load_goemotions
            train_records = load_goemotions(split="train", four_class=True)
            val_records = load_goemotions(split="validation", four_class=True)

            train_texts = [r.text for r in train_records]
            train_labels = [r.emotion_4class for r in train_records]
            val_texts = [r.text for r in val_records]
            val_labels = [r.emotion_4class for r in val_records]

            train_ds = TextEmotionDataset(train_texts, train_labels, model._tokenizer)
            val_ds = TextEmotionDataset(val_texts, val_labels, model._tokenizer)

            train_loader = DataLoader(train_ds, batch_size=config.pipeline_a.goemotion_batch_size, shuffle=True, num_workers=0, pin_memory=config.data.pin_memory)
            val_loader = DataLoader(val_ds, batch_size=config.pipeline_a.goemotion_batch_size, shuffle=False, num_workers=0, pin_memory=config.data.pin_memory)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.pipeline_a.learning_rate,
                weight_decay=config.pipeline_a.weight_decay,
            )

            total_steps = len(train_loader) * config.pipeline_a.goemotion_epochs
            scheduler = build_cosine_warmup_scheduler(optimizer, total_steps)

            loss_fn = PipelineLoss(label_smoothing=config.loss.label_smoothing)
            pipeline_a_forward._loss_fn = loss_fn

            trainer = Trainer(
                model=model, optimizer=optimizer, loss_fn=loss_fn, device=device,
                train_loader=train_loader, val_loader=val_loader,
                scheduler=scheduler, total_epochs=config.pipeline_a.goemotion_epochs,
                log_every_n_steps=config.logging.log_every_n_steps,
                forward_fn=pipeline_a_forward, num_classes=4,
                checkpoint_dir=PROJECT_ROOT / "checkpoints" / "pipeline_a" / "goemotion",
            )
            trainer.train()
            logger.info("GoEmotions pre-training complete.")

        except Exception as e:
            logger.warning(f"GoEmotions pre-training failed: {e}. Continuing with fine-tuning.")

    # ---- Step 2: IEMOCAP fine-tuning (5-fold CV) ----
    logger.info("=" * 60)
    logger.info("Step 2: IEMOCAP Fine-tuning (5-fold CV)")
    logger.info("=" * 60)

    metadata_path = PROJECT_ROOT / "data" / "processed" / "metadata.csv"
    if not metadata_path.exists():
        logger.error(f"Metadata not found: {metadata_path}. Run preprocess_all.py first.")
        sys.exit(1)

    metadata = pd.read_csv(metadata_path)
    iemocap = metadata[metadata["dataset"] == "iemocap"].reset_index(drop=True)

    if iemocap.empty:
        logger.error("No IEMOCAP entries in metadata. Check preprocessing.")
        sys.exit(1)

    # Load transcripts
    import json
    texts, labels, sessions = [], [], []
    for _, row in iemocap.iterrows():
        tpath = Path(row["transcript_path"])
        if tpath.exists():
            with open(tpath) as f:
                texts.append(json.load(f).get("text", ""))
        else:
            texts.append("")
        labels.append(row["emotion_4class"])
        # Extract session from file_id (e.g. "iemocap_Ses01F_impro01_F000" → session 1)
        fid = row["file_id"]
        try:
            sess = int(fid.split("Ses")[1][:2])
        except (IndexError, ValueError):
            sess = 1
        sessions.append(sess)

    iemocap["text"] = texts
    iemocap["session"] = sessions

    folds = range(1, 6) if args.fold is None else [args.fold + 1]

    for fold_session in folds:
        logger.info(f"\n--- Fold {fold_session} (Test=Session {fold_session}) ---")
        seed_everything(config.seed)

        train_mask = iemocap["session"] != fold_session
        val_mask = iemocap["session"] == fold_session

        train_texts = iemocap.loc[train_mask, "text"].tolist()
        train_labels = iemocap.loc[train_mask, "emotion_4class"].tolist()
        val_texts = iemocap.loc[val_mask, "text"].tolist()
        val_labels = iemocap.loc[val_mask, "emotion_4class"].tolist()

        train_ds = TextEmotionDataset(train_texts, train_labels, model._tokenizer)
        val_ds = TextEmotionDataset(val_texts, val_labels, model._tokenizer)

        class_weights = compute_class_weights(train_ds.labels.tolist())
        loss_fn = PipelineLoss(class_weights=class_weights.to(device), label_smoothing=config.loss.label_smoothing)
        pipeline_a_forward._loss_fn = loss_fn

        train_loader = DataLoader(train_ds, batch_size=config.pipeline_a.finetune_batch_size, shuffle=True, num_workers=0, pin_memory=config.data.pin_memory)
        val_loader = DataLoader(val_ds, batch_size=config.pipeline_a.finetune_batch_size, shuffle=False, num_workers=0, pin_memory=config.data.pin_memory)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.pipeline_a.learning_rate,
            weight_decay=config.pipeline_a.weight_decay,
        )
        total_steps = len(train_loader) * config.pipeline_a.finetune_epochs
        scheduler = build_cosine_warmup_scheduler(optimizer, total_steps)

        ckpt_dir = PROJECT_ROOT / "checkpoints" / "pipeline_a" / f"fold_{fold_session}"
        es = EarlyStopping(
            patience=config.early_stopping.patience,
            metric="ua", mode="max",
            checkpoint_path=ckpt_dir / "best.pt",
        )

        trainer = Trainer(
            model=model, optimizer=optimizer, loss_fn=loss_fn, device=device,
            train_loader=train_loader, val_loader=val_loader,
            scheduler=scheduler, total_epochs=config.pipeline_a.finetune_epochs,
            early_stopping=es, forward_fn=pipeline_a_forward, num_classes=4,
            checkpoint_dir=ckpt_dir, log_every_n_steps=config.logging.log_every_n_steps,
        )
        results = trainer.train()
        logger.info(f"Fold {fold_session} best UA: {results['best_metric']:.4f}")

    logger.info("Pipeline A training complete.")


if __name__ == "__main__":
    main()
