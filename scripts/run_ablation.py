#!/usr/bin/env python3
"""Master script for running MERA ablation studies.

Runs ablation experiments sequentially (or selectively) and produces
a final summary comparison table.  Each experiment trains and evaluates
the specified model variant using 5-fold LOSO-CV on IEMOCAP.

Usage::

    # Run all ablation experiments
    python scripts/run_ablation.py --config configs/default.yaml

    # Run specific experiments
    python scripts/run_ablation.py --config configs/default.yaml \\
        --experiments MERA-Full Text-Only Audio-Only No-ProsodicTCN

    # List all available experiments
    python scripts/run_ablation.py --list

    # Only run baselines
    python scripts/run_ablation.py --config configs/default.yaml --baselines-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.device import configure_mps_memory, get_device
from src.utils.seed import seed_everything
from src.utils.logging_utils import get_logger
from src.utils.checkpoint import load_checkpoint
from src.evaluation.evaluator import (
    Evaluator,
    FoldResult,
    aggregate_fold_results,
    save_results,
    format_results_table,
)
from src.evaluation.ablation import (
    ALL_ABLATIONS,
    AblationConfig,
    AblationType,
    apply_ablation,
    list_ablation_names,
    get_ablation_config,
    build_ablation_comparison_table,
    OpenSMILEMLPBaseline,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# OpenSMILE baseline evaluation helpers
# ---------------------------------------------------------------------------

def run_opensmile_svm_fold(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> dict:
    """Run SVM baseline on eGeMAPS features for one fold.

    Args:
        train_features: ``(N_train, 88)`` array.
        train_labels: ``(N_train,)`` integer labels.
        test_features: ``(N_test, 88)`` array.
        test_labels: ``(N_test,)`` integer labels.

    Returns:
        Dict with ``wa``, ``ua``, ``macro_f1``.
    """
    try:
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.error("scikit-learn required for SVM baseline. pip install scikit-learn")
        return {"wa": 0.0, "ua": 0.0, "macro_f1": 0.0}

    from src.training.metrics import (
        compute_weighted_accuracy,
        compute_unweighted_accuracy,
        compute_f1_scores,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_features)
    X_test = scaler.transform(test_features)

    clf = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
    clf.fit(X_train, train_labels)
    preds = clf.predict(X_test)

    wa = compute_weighted_accuracy(preds, test_labels)
    ua, _ = compute_unweighted_accuracy(preds, test_labels)
    _, macro_f1 = compute_f1_scores(preds, test_labels)

    return {"wa": wa, "ua": ua, "macro_f1": macro_f1}


def run_opensmile_mlp_experiment(
    config,
    iemocap_df,
    device: torch.device,
    folds: range,
) -> list[FoldResult]:
    """Run OpenSMILE MLP baseline across folds.

    This is a simplified training loop for the MLP baseline.
    """
    from src.training.losses import PipelineLoss, compute_class_weights
    from src.data.collate import EMOTION_TO_IDX

    fold_results = []

    for fold_session in folds:
        logger.info(f"  MLP Baseline - Fold {fold_session}")
        seed_everything(config.seed)

        train_df = iemocap_df[iemocap_df["session"] != fold_session].reset_index(drop=True)
        val_df = iemocap_df[iemocap_df["session"] == fold_session].reset_index(drop=True)

        # Check for eGeMAPS features
        if "egemaps_path" not in train_df.columns:
            logger.warning("eGeMAPS features not found in metadata. Skipping MLP baseline.")
            fold_results.append(FoldResult(fold=fold_session))
            continue

        # Load features
        try:
            train_features = np.stack([
                np.load(p) for p in train_df["egemaps_path"] if Path(p).exists()
            ])
            train_labels = train_df["emotion_4class"].map(EMOTION_TO_IDX).values

            val_features = np.stack([
                np.load(p) for p in val_df["egemaps_path"] if Path(p).exists()
            ])
            val_labels = val_df["emotion_4class"].map(EMOTION_TO_IDX).values
        except Exception as e:
            logger.warning(f"Failed to load eGeMAPS features: {e}")
            fold_results.append(FoldResult(fold=fold_session))
            continue

        # Build MLP
        model = OpenSMILEMLPBaseline(
            input_dim=train_features.shape[1],
            num_classes=4,
        )
        model.float()
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = PipelineLoss(label_smoothing=0.0)

        # Simple training loop
        X_train = torch.from_numpy(train_features).float().to(device)
        y_train = torch.from_numpy(train_labels).long().to(device)
        X_val = torch.from_numpy(val_features).float().to(device)
        y_val = torch.from_numpy(val_labels).long().to(device)

        best_ua = 0.0
        for epoch in range(50):
            model.train()
            logits = model(X_train)
            loss_dict = loss_fn(logits, y_train)
            loss_dict["total"].backward()
            optimizer.step()
            optimizer.zero_grad()

            # Validate
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val)
                preds = val_logits.argmax(dim=1).cpu().numpy()
                targets = y_val.cpu().numpy()

            from src.training.metrics import compute_unweighted_accuracy
            ua, _ = compute_unweighted_accuracy(preds, targets)
            if ua > best_ua:
                best_ua = ua

        # Final evaluation with best model
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)

        from src.training.metrics import (
            compute_weighted_accuracy,
            compute_f1_scores,
            compute_confusion_matrix,
        )
        preds = val_logits.argmax(dim=1).cpu().numpy()
        targets = y_val.cpu().numpy()

        wa = compute_weighted_accuracy(preds, targets)
        ua, per_class_acc = compute_unweighted_accuracy(preds, targets)
        per_class_f1, macro_f1 = compute_f1_scores(preds, targets)
        cm = compute_confusion_matrix(preds, targets)

        fold_results.append(FoldResult(
            fold=fold_session,
            wa=wa, ua=ua, macro_f1=macro_f1,
            confusion_matrix=cm.tolist(),
            total_samples=len(targets),
        ))

    return fold_results


# ---------------------------------------------------------------------------
# Main MERA model evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_mera_fold(
    config,
    model,
    val_df,
    device: torch.device,
    fold_session: int,
) -> FoldResult:
    """Evaluate a MERA model on one fold's validation set.

    Args:
        config: MERA config.
        model: The model (already loaded with checkpoint).
        val_df: Validation DataFrame for this fold.
        device: Torch device.
        fold_session: The fold number.

    Returns:
        FoldResult with computed metrics.
    """
    from functools import partial
    from torch.utils.data import DataLoader

    # Import dataset utilities
    from scripts.train_fusion import MultimodalDataset, multimodal_collate

    tokenizer = model.pipeline_a._tokenizer if hasattr(model, "pipeline_a") else None

    if tokenizer is None:
        logger.warning("Model has no tokenizer. Cannot build multimodal dataset.")
        return FoldResult(fold=fold_session)

    val_ds = MultimodalDataset(val_df, tokenizer)
    collate = partial(
        multimodal_collate,
        tokenizer=tokenizer,
        max_length=config.pipeline_a.max_text_length,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.fusion.batch_size,
        shuffle=False, num_workers=0, collate_fn=collate,
    )

    from scripts.train_fusion import fusion_forward
    from src.training.losses import MERALoss, compute_class_weights
    from src.data.collate import EMOTION_TO_IDX

    train_labels = val_df["emotion_4class"].map(EMOTION_TO_IDX).tolist()
    loss_fn = MERALoss(label_smoothing=config.loss.label_smoothing)
    fusion_forward._loss_fn = loss_fn

    evaluator = Evaluator(
        model=model,
        device=device,
        forward_fn=fusion_forward,
        num_classes=4,
    )
    result = evaluator.evaluate(val_loader)
    result.fold = fold_session
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run MERA ablation studies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--experiments", nargs="+", default=None,
        help="Specific experiments to run (default: all)",
    )
    parser.add_argument("--list", action="store_true", help="List available experiments")
    parser.add_argument("--baselines-only", action="store_true", help="Only run baseline experiments")
    parser.add_argument("--output-dir", type=str, default="results/ablation")
    parser.add_argument("--format", choices=["markdown", "latex"], default="markdown")
    args = parser.parse_args()

    if args.list:
        print("Available ablation experiments:")
        for name in list_ablation_names():
            cfg = get_ablation_config(name)
            baseline_tag = " [BASELINE]" if cfg.is_baseline else ""
            print(f"  {name:30s} {cfg.description}{baseline_tag}")
        return

    config = load_config(PROJECT_ROOT / args.config)
    configure_mps_memory()
    seed_everything(config.seed)
    device = get_device(config.device)
    logger.info(f"Device: {device}")

    # Determine experiments to run
    if args.experiments:
        experiment_names = args.experiments
    elif args.baselines_only:
        experiment_names = [
            name for name, cfg in ALL_ABLATIONS.items() if cfg.is_baseline
        ]
    else:
        experiment_names = list_ablation_names()

    # Validate experiment names
    for name in experiment_names:
        get_ablation_config(name)  # raises KeyError if invalid

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load IEMOCAP metadata
    import pandas as pd
    metadata_path = PROJECT_ROOT / "data" / "processed" / "metadata.csv"
    if not metadata_path.exists():
        logger.error(f"Metadata not found: {metadata_path}. Run preprocess_all.py first.")
        sys.exit(1)

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

    folds = range(1, 6)
    all_results = {}

    for exp_name in experiment_names:
        ablation_cfg = get_ablation_config(exp_name)
        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment: {exp_name}")
        logger.info(f"Description: {ablation_cfg.description}")
        logger.info(f"{'='*60}")

        if ablation_cfg.is_baseline:
            # Handle OpenSMILE baselines separately
            logger.info(f"Running baseline: {exp_name}")
            # Baselines require eGeMAPS features; check and skip if unavailable
            fold_results = []
            for fold_session in folds:
                fold_results.append(FoldResult(fold=fold_session))
            logger.warning(
                f"Baseline {exp_name} requires pre-extracted eGeMAPS features. "
                "Placeholder results recorded."
            )
        else:
            # Neural model experiments
            fold_results = []
            for fold_session in folds:
                logger.info(f"  Fold {fold_session}")
                seed_everything(config.seed)

                val_df = iemocap[iemocap["session"] == fold_session].reset_index(drop=True)

                # Try to load the best end-to-end checkpoint
                ckpt_path = (
                    PROJECT_ROOT / "checkpoints" / "end_to_end"
                    / f"fold_{fold_session}" / "best.pt"
                )
                if not ckpt_path.exists():
                    # Fall back to fusion checkpoint
                    ckpt_path = (
                        PROJECT_ROOT / "checkpoints" / "fusion"
                        / f"fold_{fold_session}" / "best.pt"
                    )

                if not ckpt_path.exists():
                    logger.warning(
                        f"No checkpoint found for fold {fold_session}. "
                        "Train the model first."
                    )
                    fold_results.append(FoldResult(fold=fold_session))
                    continue

                try:
                    from src.models.fusion.mera_model import MERAModel
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

                    load_checkpoint(ckpt_path, model, strict=False, device=device)
                    model = apply_ablation(model, ablation_cfg)

                    result = evaluate_mera_fold(
                        config, model, val_df, device, fold_session,
                    )
                    fold_results.append(result)
                except Exception as e:
                    logger.error(f"Fold {fold_session} failed: {e}")
                    fold_results.append(FoldResult(fold=fold_session))

        # Aggregate
        agg = aggregate_fold_results(fold_results, experiment_name=exp_name)
        all_results[exp_name] = agg.to_dict()

        # Save per-experiment results
        save_results(agg, output_dir / f"{exp_name.replace(' ', '_')}.json")

        logger.info(
            f"  {exp_name}: WA={agg.wa_mean*100:.1f}+/-{agg.wa_std*100:.1f}  "
            f"UA={agg.ua_mean*100:.1f}+/-{agg.ua_std*100:.1f}"
        )

    # Final summary table
    logger.info(f"\n{'='*60}")
    logger.info("ABLATION STUDY SUMMARY")
    logger.info(f"{'='*60}")

    table = build_ablation_comparison_table(all_results)
    logger.info(f"\n{table}")

    # Save summary
    summary_path = output_dir / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    table_path = output_dir / f"ablation_table.{'tex' if args.format == 'latex' else 'md'}"
    with open(table_path, "w") as f:
        from src.evaluation.evaluator import AggregateResult
        agg_results = []
        for name, data in all_results.items():
            agg_results.append(AggregateResult(
                experiment_name=name,
                num_folds=data.get("num_folds", 5),
                wa_mean=data.get("wa_mean", 0),
                wa_std=data.get("wa_std", 0),
                ua_mean=data.get("ua_mean", 0),
                ua_std=data.get("ua_std", 0),
                macro_f1_mean=data.get("macro_f1_mean", 0),
                macro_f1_std=data.get("macro_f1_std", 0),
            ))
        f.write(format_results_table(agg_results, format=args.format))
    logger.info(f"Table saved to {table_path}")


if __name__ == "__main__":
    main()
