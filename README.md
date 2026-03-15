# MERA: Multimodal Emotion Recognition Architecture

**Cross-Modal Gated Attention Fusion for Speech Emotion Recognition**

*CS 298 Master's Thesis — San Jose State University*

---

## Overview

MERA is a late-fusion multimodal system for speech emotion recognition (SER) that combines linguistic (text) and paralinguistic (audio) cues through a novel cross-modal gated attention fusion mechanism. The system classifies utterances into four emotion categories: **angry**, **happy**, **sad**, and **neutral**.

### Thesis Contributions

1. **Prosodic Contour TCN**: Frame-level 10-channel prosodic contours processed by dilated causal temporal convolutional networks, replacing traditional summary statistics with fine-grained temporal dynamics.

2. **Attention Entropy as Linguistic Confidence**: Shannon entropy of DeBERTa's attention distributions serves as an explicit confidence signal that informs the fusion gate — high entropy indicates linguistically ambiguous text, shifting reliance toward audio cues.

3. **Modality Dropout Consistency Training**: During training, individual modalities are randomly zeroed with a KL-divergence consistency loss, ensuring robustness when either modality is degraded.

4. **Three-Stream Paralinguistic Pipeline with SE Weighting**: CNN-BiLSTM (spectrograms), Prosodic TCN (contours), and HuBERT (self-supervised embeddings) are fused via Squeeze-and-Excitation channel reweighting.

---

## Architecture

```
Raw Audio (.wav, 16kHz mono)
        |
        +-- [Whisper large-v3] ---------> Transcript text
        |                                      |
        |                            [DeBERTa-v3-base]
        |                                      |
        |                         +------------+------------+
        |                         |                         |
        |                  Text logits (4-d)     Attention Entropy (12-d)
        |                         |                         |
        |                   [Pipeline A]             [Confidence Signal]
        |                         |                         |
        +-- [Log-Mel Spectrogram] --> [CNN-BiLSTM] --> Stream 1 (256-d)    |
        |                                                   |              |
        +-- [Prosodic Contours]   --> [TCN]        --> Stream 2 (128-d)    |
        |                                                   |              |
        +-- [HuBERT-Large frozen] --> [Weighted Sum]--> Stream 3 (256-d)   |
                                                        |              |
                                            [Squeeze-and-Excitation]   |
                                                        |              |
                                                  Audio repr (256-d)   |
                                                        |              |
                                                  [Pipeline B]         |
                                                        |              |
                              +-------------------------+              |
                              |                                        |
                    [Cross-Modal Attention]                             |
                     (bidirectional)                                    |
                              |                                        |
                    [Gated Fusion] <-----------------------------------+
                     (entropy-informed gate)
                              |
                    Final Emotion Prediction (4-d)
                    {angry, happy, sad, neutral}
```

---

## Installation

### Requirements

- Python 3.10+
- Apple Silicon Mac (M1/M2/M3/M4/M5 Pro) with 24 GB unified memory recommended
- PyTorch 2.0+ with MPS backend support

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/cs298multimodal.git
cd cs298multimodal

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify MPS availability
python3 -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### Key Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Deep learning framework (MPS backend) |
| `transformers` | DeBERTa-v3-base, HuBERT-Large |
| `mlx-whisper` | Whisper transcription (Apple Silicon optimised) |
| `librosa` | Audio loading and processing |
| `praat-parselmouth` | Prosodic feature extraction (F0, HNR, jitter, shimmer, formants) |
| `soundfile` | Audio I/O |
| `gradio` | Interactive demo UI |
| `wandb` | Experiment tracking (optional) |

---

## Dataset Setup

### IEMOCAP (Primary)

IEMOCAP requires a licence from USC. After obtaining access:

```bash
mkdir -p data/raw/iemocap
# Place the IEMOCAP session directories here:
# data/raw/iemocap/Session1/
# data/raw/iemocap/Session2/
# ...
# data/raw/iemocap/Session5/
```

### Optional Datasets

| Dataset | Purpose | Location |
|---------|---------|----------|
| RAVDESS | Additional training data | `data/raw/ravdess/` |
| CREMA-D | Additional training data | `data/raw/cremad/` |
| MSP-IMPROV | Cross-dataset evaluation | `data/raw/msp_improv/` |
| GoEmotions | Text pre-training (auto-downloaded) | Via HuggingFace |

---

## Preprocessing

Extract all features from raw audio:

```bash
# Full preprocessing pipeline (Whisper -> Spectrogram -> Prosodic -> HuBERT)
python scripts/preprocess_all.py --config configs/default.yaml

# Skip stages that are already done
python scripts/preprocess_all.py --config configs/default.yaml --skip-whisper --skip-hubert

# Process a single dataset
python scripts/preprocess_all.py --config configs/default.yaml --dataset iemocap
```

This creates:
- `data/processed/metadata.csv` — master index of all utterances
- `data/processed/transcripts/` — Whisper JSON transcripts
- `data/processed/spectrograms/` — 128-bin log-mel spectrograms (.npy)
- `data/processed/prosodic/` — 10-channel prosodic contours (.npy)
- `data/processed/hubert/` — 25-layer HuBERT embeddings (.npy, float16)

---

## Training Pipeline

MERA is trained in four stages, each building on the previous:

### Stage 1: Pipeline A (Text)

Pre-train on GoEmotions, then fine-tune on IEMOCAP transcripts:

```bash
python scripts/train_pipeline_a.py --config configs/default.yaml

# Skip GoEmotions pre-training
python scripts/train_pipeline_a.py --config configs/default.yaml --skip-goemotion

# Single fold only
python scripts/train_pipeline_a.py --config configs/default.yaml --fold 0
```

### Stage 2: Pipeline B (Audio)

Train the three-stream audio pipeline on cached features:

```bash
python scripts/train_pipeline_b.py --config configs/default.yaml
```

### Stage 3: Fusion

Freeze both pipelines, train cross-modal attention + gated fusion:

```bash
python scripts/train_fusion.py --config configs/default.yaml
```

### Stage 4: End-to-End Fine-tuning

Unfreeze last 2 DeBERTa layers + all Pipeline B + fusion, fine-tune with gradient checkpointing:

```bash
python scripts/train_end_to_end.py --config configs/default.yaml
```

All stages use 5-fold leave-one-session-out cross-validation. Checkpoints are saved to `checkpoints/{stage}/fold_{n}/best.pt`.

---

## Evaluation

### Standard Evaluation

Metrics reported: Weighted Accuracy (WA), Unweighted Accuracy (UA), per-class F1, and confusion matrix. All results are mean +/- std across 5 folds.

### Ablation Studies

Run all 14 ablation experiments:

```bash
# Run all ablations
python scripts/run_ablation.py --config configs/default.yaml

# Run specific experiments
python scripts/run_ablation.py --experiments MERA-Full Text-Only Audio-Only No-ProsodicTCN

# List available experiments
python scripts/run_ablation.py --list

# Generate LaTeX table
python scripts/run_ablation.py --config configs/default.yaml --format latex
```

**Ablation experiments:**

| Category | Experiments |
|----------|-------------|
| Full system | MERA-Full |
| Pipeline ablations | Text-Only, Audio-Only |
| Sub-stream ablations | No-CNN-BiLSTM, No-ProsodicTCN, No-HuBERT, TCN-Only |
| Novel components | No-AttentionEntropy, No-ModalityDropout, No-CrossModalAttention, No-GatedFusion |
| Baselines | OpenSMILE-SVM, OpenSMILE-MLP, Summary-Stats-TCN |

Results are saved to `results/ablation/` as JSON and formatted tables.

---

## Inference

### Single File

```bash
python scripts/run_inference.py --audio path/to/speech.wav

# With specific checkpoint
python scripts/run_inference.py --audio speech.wav \
    --checkpoint checkpoints/end_to_end/fold_1/best.pt

# Save result as JSON
python scripts/run_inference.py --audio speech.wav --output result.json

# Force CPU device
python scripts/run_inference.py --audio speech.wav --device cpu
```

**Output:**
```
============================================================
MERA Emotion Recognition Result
============================================================
  Audio:       speech.wav
  Transcript:  "I can't believe you did that"
  Emotion:     ANGRY
  Confidence:  87.3%

  Probabilities:
     angry: 0.873  ###################################
     happy: 0.052  ##
       sad: 0.028  #
   neutral: 0.047  #

  Gate weights: text=0.350, audio=0.650
  Attention entropy (mean): 1.230

  Interpretation: High confidence (87%) prediction of 'angry'.
  Model relied more on vocal cues than text content.
============================================================
```

### Interactive Demo

```bash
# Launch Gradio demo (opens in browser)
python scripts/demo.py

# With public link
python scripts/demo.py --share

# Custom port
python scripts/demo.py --port 8080
```

The demo provides an interpretability dashboard with:
- Emotion probability bar chart
- Gate weight visualization (text vs audio contribution)
- Attention entropy heatmap across DeBERTa's 12 layers
- Prosodic contour plots (F0, energy, spectral centroid)
- Log-mel spectrogram display

---

## Apple Silicon Optimisation Notes

MERA is designed to run entirely on Apple Silicon Macs with 24 GB unified memory:

| Training Stage | Memory Usage | Notes |
|----------------|-------------|-------|
| Pipeline A | ~6 GB | DeBERTa-v3-base with frozen bottom 6 layers |
| Pipeline B | ~4 GB | Three streams + SE fusion |
| Fusion | ~6 GB | Both pipelines frozen |
| End-to-End | ~16-18 GB | Gradient checkpointing enabled |
| Inference | ~8 GB | All models loaded, no gradients |

**Key optimisations:**
- MPS memory watermarks: high=0.7, low=0.5 (via `PYTORCH_MPS_HIGH_WATERMARK_RATIO`)
- `torch.mps.empty_cache()` called between heavy operations
- HuBERT embeddings saved as float16, memory-mapped at load time
- Gradient checkpointing for end-to-end fine-tuning
- Physical batch 8 x accumulation 4 = effective batch 32
- **Important**: Always call `model.float()` before moving to MPS to avoid dtype mismatch errors (DeBERTa loads some weights as float16)

---

## Project Structure

```
cs298multimodal/
+-- configs/
|   +-- default.yaml              # All hyperparameters
+-- data/
|   +-- raw/                      # Raw dataset files
|   +-- processed/                # Extracted features + metadata
+-- src/
|   +-- data/                     # Dataset loaders, collation, augmentation
|   |   +-- iemocap_loader.py
|   |   +-- ravdess_loader.py
|   |   +-- cremad_loader.py
|   |   +-- msp_improv_loader.py
|   |   +-- goemotions_loader.py
|   |   +-- dataset_registry.py
|   |   +-- audio_utils.py
|   |   +-- collate.py
|   |   +-- augmentation.py
|   +-- features/                 # Feature extraction
|   |   +-- whisper_transcriber.py
|   |   +-- spectrogram.py
|   |   +-- prosodic.py           # NOVEL: 10-channel prosodic contours
|   |   +-- hubert_extractor.py
|   |   +-- opensmile_baseline.py
|   +-- models/
|   |   +-- pipeline_a/           # Text pipeline
|   |   |   +-- text_encoder.py
|   |   |   +-- attention_entropy.py
|   |   |   +-- text_emotion_head.py
|   |   +-- pipeline_b/           # Audio pipeline
|   |   |   +-- cnn_bilstm.py
|   |   |   +-- prosodic_tcn.py   # NOVEL: Dilated causal TCN
|   |   |   +-- hubert_head.py
|   |   |   +-- squeeze_excitation.py
|   |   |   +-- audio_emotion_head.py
|   |   +-- fusion/               # Fusion module
|   |       +-- cross_modal_attention.py
|   |       +-- gated_fusion.py   # NOVEL: Entropy-informed gate
|   |       +-- mera_model.py
|   +-- training/                 # Training infrastructure
|   |   +-- losses.py
|   |   +-- metrics.py
|   |   +-- schedulers.py
|   |   +-- early_stopping.py
|   |   +-- trainer.py
|   +-- evaluation/               # Evaluation & ablation
|   |   +-- evaluator.py
|   |   +-- ablation.py
|   |   +-- cross_dataset.py
|   +-- utils/                    # Utilities
|       +-- config.py
|       +-- device.py
|       +-- seed.py
|       +-- logging_utils.py
|       +-- checkpoint.py
+-- scripts/
|   +-- preprocess_all.py         # Feature extraction pipeline
|   +-- train_pipeline_a.py       # Stage 1: Text
|   +-- train_pipeline_b.py       # Stage 2: Audio
|   +-- train_fusion.py           # Stage 3: Fusion
|   +-- train_end_to_end.py       # Stage 4: End-to-end
|   +-- run_ablation.py           # Ablation studies
|   +-- run_inference.py          # Single-file inference
|   +-- demo.py                   # Gradio interactive demo
+-- tests/
|   +-- test_data_loaders.py      # Phase 1 tests (35)
|   +-- test_feature_extraction.py # Phase 2 tests (35)
|   +-- test_models.py            # Phase 3 tests (36)
|   +-- test_training.py          # Phase 4 tests (48)
|   +-- test_evaluation.py        # Phase 5 tests (59)
|   +-- test_inference.py         # Phase 6 tests
+-- checkpoints/                  # Saved model checkpoints
+-- results/                      # Evaluation results
+-- requirements.txt
+-- README.md
```

---

## Running Tests

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run specific phase tests
python3 -m pytest tests/test_data_loaders.py -v       # Phase 1
python3 -m pytest tests/test_feature_extraction.py -v  # Phase 2
python3 -m pytest tests/test_models.py -v              # Phase 3
python3 -m pytest tests/test_training.py -v            # Phase 4
python3 -m pytest tests/test_evaluation.py -v          # Phase 5
python3 -m pytest tests/test_inference.py -v           # Phase 6
```

---

## Configuration

All hyperparameters are in `configs/default.yaml`. Key settings:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `seed` | 42 | Random seed for reproducibility |
| `pipeline_a.model_name` | `microsoft/deberta-v3-base` | Text encoder |
| `pipeline_b.tcn_blocks` | 6 | Prosodic TCN depth (receptive field: 2.52s) |
| `fusion.embed_dim` | 256 | Shared representation dimension |
| `fusion.modality_dropout_prob` | 0.3 | Modality dropout probability |
| `end_to_end.learning_rate` | 5e-5 | Fine-tuning LR |
| `end_to_end.gradient_accumulation` | 4 | Effective batch = 8 x 4 = 32 |
| `loss.lambda_consistency` | 0.2 | KL consistency loss weight |
| `early_stopping.patience` | 7 | Epochs before stopping |

---

## Citation

If you use MERA in your research, please cite:

```bibtex
@mastersthesis{munshi2026mera,
  title={MERA: Cross-Modal Gated Attention Fusion for Multimodal Speech Emotion Recognition},
  author={Munshi, Shubhankar},
  year={2026},
  school={San Jose State University},
  type={Master's Thesis}
}
```

---

## Licence

This project is part of a Master's thesis at San Jose State University. Please contact the author for usage permissions.
