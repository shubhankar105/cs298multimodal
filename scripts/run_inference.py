#!/usr/bin/env python3
"""Single-file inference pipeline for the MERA system.

Loads a trained MERA checkpoint, processes a single audio file through
the full pipeline (Whisper transcription -> feature extraction -> model
inference), and outputs emotion prediction with interpretability info.

Usage::

    python scripts/run_inference.py --audio path/to/audio.wav
    python scripts/run_inference.py --audio path/to/audio.wav --checkpoint checkpoints/end_to_end/fold_1/best.pt
    python scripts/run_inference.py --audio path/to/audio.wav --output result.json --device cpu

Output includes:
- Predicted emotion with confidence score
- Probability distribution across all 4 classes
- Transcript text
- Gate weights (text vs audio reliance)
- Mean attention entropy
- Human-readable interpretation string
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Emotion labels
EMOTION_LABELS = ["angry", "happy", "sad", "neutral"]


class MERAInference:
    """Full MERA inference pipeline.

    Loads a trained checkpoint, runs Whisper transcription, extracts
    audio features (spectrogram, prosodic contours, HuBERT embeddings),
    tokenises the transcript, runs the MERA model, and returns a rich
    result dict with predictions and interpretability information.

    Args:
        checkpoint_path: Path to the trained MERA ``best.pt`` checkpoint.
        config_path: Path to the YAML configuration file.
        device: ``"mps"``, ``"cpu"``, or ``"cuda"``.
        whisper_model_id: HuggingFace Whisper model identifier.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path = "configs/default.yaml",
        device: str = "auto",
        whisper_model_id: Optional[str] = None,
    ):
        from src.utils.config import load_config
        from src.utils.device import get_device
        from src.utils.checkpoint import load_checkpoint
        from src.models.fusion.mera_model import MERAModel

        self.config = load_config(PROJECT_ROOT / config_path)

        # Device selection
        if device == "auto":
            self.device = get_device()
        else:
            self.device = torch.device(device)

        logger.info(f"Inference device: {self.device}")

        # Build model
        self.model = MERAModel(
            num_classes=4,
            text_model_name=self.config.pipeline_a.model_name,
            embed_dim=self.config.fusion.embed_dim,
            dropout=self.config.pipeline_a.dropout,
            modality_dropout_prob=0.0,  # No dropout at inference
            tcn_channels=self.config.pipeline_b.tcn_channels,
            tcn_blocks=self.config.pipeline_b.tcn_blocks,
            tcn_kernel_size=self.config.pipeline_b.tcn_kernel_size,
            prosodic_output_dim=self.config.pipeline_b.prosodic_output_dim,
            se_reduction=self.config.pipeline_b.se_reduction,
            cross_attention_heads=self.config.fusion.cross_attention_heads,
        )

        # IMPORTANT: Ensure all parameters are float32 before moving to MPS.
        # DeBERTa loads some weights as float16 which causes MPS dtype errors.
        self.model.float()
        self.model.to(self.device)

        # Load checkpoint
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            load_checkpoint(checkpoint_path, self.model, strict=False, device=self.device)
            logger.info(f"Checkpoint loaded from {checkpoint_path}")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}. Using uninitialised weights.")

        self.model.eval()

        # Tokenizer (from the DeBERTa encoder)
        self._tokenizer = self.model.pipeline_a._tokenizer

        # Whisper backend and model ID
        self._whisper_backend = self.config.whisper_backend
        if whisper_model_id:
            self._whisper_model_id = whisper_model_id
        elif self._whisper_backend == "huggingface":
            self._whisper_model_id = "openai/whisper-large-v3"
        else:
            self._whisper_model_id = "mlx-community/whisper-large-v3-mlx"

    def predict(self, audio_path: str | Path) -> dict:
        """Run full inference on a single audio file.

        Args:
            audio_path: Path to the input audio file (WAV, FLAC, MP3, etc.).

        Returns:
            Dict with keys:
            - ``emotion``: predicted emotion string
            - ``confidence``: confidence score (max probability)
            - ``probabilities``: dict mapping emotion names to probabilities
            - ``transcript``: Whisper transcript text
            - ``gate_weights``: dict with ``text`` and ``audio`` weights
            - ``attention_entropy``: per-layer entropy list
            - ``attention_entropy_mean``: mean attention entropy
            - ``interpretation``: human-readable interpretation string

        Raises:
            FileNotFoundError: If the audio file does not exist.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Processing: {audio_path}")

        # --- Step 1: Load and preprocess audio ---
        from src.data.audio_utils import load_and_preprocess
        audio = load_and_preprocess(
            audio_path,
            sr=self.config.data.sample_rate,
            max_duration_sec=self.config.data.max_audio_duration_sec,
        )
        logger.info(f"Audio loaded: {len(audio)/self.config.data.sample_rate:.2f}s")

        # --- Step 2: Transcribe with Whisper ---
        transcript = self._transcribe(audio_path)
        logger.info(f"Transcript: {transcript}")

        # --- Step 3: Extract features ---
        spectrogram, prosody, hubert = self._extract_features(audio)

        # --- Step 4: Tokenise transcript ---
        encoded = self._tokenizer(
            transcript,
            padding=True,
            truncation=True,
            max_length=self.config.pipeline_a.max_text_length,
            return_tensors="pt",
        )

        # --- Step 5: Run model ---
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoded["input_ids"].to(self.device),
                attention_mask=encoded["attention_mask"].to(self.device),
                spectrogram=spectrogram.unsqueeze(0).to(self.device),
                spec_mask=torch.ones(1, spectrogram.shape[-1], dtype=torch.bool, device=self.device),
                prosody=prosody.unsqueeze(0).to(self.device),
                pros_mask=torch.ones(1, prosody.shape[-1], dtype=torch.bool, device=self.device),
                hubert=hubert.unsqueeze(0).to(self.device),
                hub_mask=torch.ones(1, hubert.shape[1], dtype=torch.bool, device=self.device),
            )

        # --- Step 6: Interpret results ---
        return self._build_result(outputs, transcript)

    def predict_from_features(
        self,
        transcript: str,
        spectrogram: torch.Tensor,
        prosody: torch.Tensor,
        hubert: torch.Tensor,
    ) -> dict:
        """Run inference from pre-extracted features.

        Useful for evaluation when features are already cached.

        Args:
            transcript: The text transcript.
            spectrogram: ``(128, T)`` log-mel spectrogram tensor.
            prosody: ``(10, T)`` prosodic contour tensor.
            hubert: ``(25, T_h, 1024)`` HuBERT embedding tensor.

        Returns:
            Same result dict as ``predict()``.
        """
        encoded = self._tokenizer(
            transcript,
            padding=True,
            truncation=True,
            max_length=self.config.pipeline_a.max_text_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=encoded["input_ids"].to(self.device),
                attention_mask=encoded["attention_mask"].to(self.device),
                spectrogram=spectrogram.unsqueeze(0).to(self.device),
                spec_mask=torch.ones(1, spectrogram.shape[-1], dtype=torch.bool, device=self.device),
                prosody=prosody.unsqueeze(0).to(self.device),
                pros_mask=torch.ones(1, prosody.shape[-1], dtype=torch.bool, device=self.device),
                hubert=hubert.unsqueeze(0).to(self.device),
                hub_mask=torch.ones(1, hubert.shape[1], dtype=torch.bool, device=self.device),
            )

        return self._build_result(outputs, transcript)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transcribe(self, audio_path: Path) -> str:
        """Transcribe audio using Whisper.

        Automatically selects backend based on config (``mlx`` or ``huggingface``).
        Falls back to empty string if Whisper is unavailable.
        """
        try:
            if self._whisper_backend == "huggingface":
                from src.features.whisper_transcriber import _transcribe_huggingface
                result = _transcribe_huggingface(
                    str(audio_path), self._whisper_model_id, "en",
                )
            else:
                import mlx_whisper
                result = mlx_whisper.transcribe(
                    str(audio_path),
                    path_or_hf_repo=self._whisper_model_id,
                    language="en",
                    fp16=True,
                )
            return result.get("text", "").strip()
        except ImportError as exc:
            logger.warning(
                f"Whisper backend '{self._whisper_backend}' not available ({exc}). "
                "Falling back to empty transcript."
            )
            return ""
        except Exception as e:
            logger.warning(f"Whisper transcription failed: {e}. Using empty transcript.")
            return ""

    def _extract_features(
        self,
        audio: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract spectrogram, prosodic contours, and HuBERT embeddings.

        Args:
            audio: Preprocessed 1-D float32 audio waveform (16 kHz).

        Returns:
            Tuple of (spectrogram, prosody, hubert) tensors.
        """
        from src.features.spectrogram import extract_log_mel
        from src.features.prosodic import extract_prosodic_contours

        # Spectrogram
        spec = extract_log_mel(audio, sr=self.config.data.sample_rate)
        spectrogram = torch.from_numpy(spec).float()

        # Prosodic contours
        pros = extract_prosodic_contours(audio, sr=self.config.data.sample_rate)
        prosody = torch.from_numpy(pros).float()

        # HuBERT embeddings
        hubert = self._extract_hubert_single(audio)

        return spectrogram, prosody, hubert

    def _extract_hubert_single(self, audio: np.ndarray) -> torch.Tensor:
        """Extract HuBERT embeddings for a single audio waveform.

        Args:
            audio: 1-D float32 audio waveform (16 kHz).

        Returns:
            ``(25, T, 1024)`` float32 tensor.
        """
        try:
            from transformers import HubertModel, Wav2Vec2FeatureExtractor

            if not hasattr(self, "_hubert_model"):
                logger.info("Loading HuBERT model for inference...")
                self._hubert_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                    "facebook/hubert-large-ls960-ft"
                )
                self._hubert_model = HubertModel.from_pretrained(
                    "facebook/hubert-large-ls960-ft",
                    output_hidden_states=True,
                )
                self._hubert_model.float()
                self._hubert_model.to(self.device)
                self._hubert_model.eval()

            inputs = self._hubert_extractor(
                audio,
                sampling_rate=self.config.data.sample_rate,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs.input_values.float().to(self.device)

            with torch.no_grad():
                outputs = self._hubert_model(input_values)

            hidden_states = outputs.hidden_states  # tuple of (1, T, 1024) x 25
            stacked = torch.stack(hidden_states, dim=1).squeeze(0)  # (25, T, 1024)
            return stacked.cpu().float()

        except ImportError:
            logger.warning(
                "transformers not available for HuBERT. "
                "Using zeros. Install with: pip install transformers"
            )
            # Return dummy zeros
            n_frames = len(audio) // 320  # HuBERT ~50 fps
            return torch.zeros(25, max(n_frames, 1), 1024)
        except Exception as e:
            logger.warning(f"HuBERT extraction failed: {e}. Using zeros.")
            n_frames = len(audio) // 320
            return torch.zeros(25, max(n_frames, 1), 1024)

    def _build_result(self, outputs: dict[str, torch.Tensor], transcript: str) -> dict:
        """Build the result dict from model outputs.

        Args:
            outputs: Dict from ``MERAModel.forward()``.
            transcript: The Whisper transcript.

        Returns:
            Rich result dict with predictions and interpretability info.
        """
        # Probabilities
        logits = outputs["final_logits"].squeeze(0)  # (4,)
        probs = F.softmax(logits, dim=0).cpu().numpy()
        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])
        predicted_emotion = EMOTION_LABELS[pred_idx]

        probabilities = {
            label: round(float(probs[i]), 4)
            for i, label in enumerate(EMOTION_LABELS)
        }

        # Gate weights
        gate_weights_raw = outputs["gate_weights"].squeeze(0).cpu().numpy()  # (2,)
        gate_weights = {
            "text": round(float(gate_weights_raw[0]), 4),
            "audio": round(float(gate_weights_raw[1]), 4),
        }

        # Attention entropy
        attention_entropy_raw = outputs["attention_entropy"].squeeze(0).cpu().numpy()  # (12,)
        attention_entropy = [round(float(e), 4) for e in attention_entropy_raw]
        entropy_mean = round(float(attention_entropy_raw.mean()), 4)

        # Human-readable interpretation
        interpretation = self._generate_interpretation(
            predicted_emotion, confidence, gate_weights, entropy_mean,
        )

        # Sub-pipeline predictions
        text_probs = F.softmax(outputs["text_logits"].squeeze(0), dim=0).cpu().numpy()
        audio_probs = F.softmax(outputs["audio_logits"].squeeze(0), dim=0).cpu().numpy()

        return {
            "emotion": predicted_emotion,
            "confidence": round(confidence, 4),
            "probabilities": probabilities,
            "transcript": transcript,
            "gate_weights": gate_weights,
            "attention_entropy": attention_entropy,
            "attention_entropy_mean": entropy_mean,
            "interpretation": interpretation,
            "text_prediction": EMOTION_LABELS[int(text_probs.argmax())],
            "audio_prediction": EMOTION_LABELS[int(audio_probs.argmax())],
            "text_probabilities": {
                label: round(float(text_probs[i]), 4)
                for i, label in enumerate(EMOTION_LABELS)
            },
            "audio_probabilities": {
                label: round(float(audio_probs[i]), 4)
                for i, label in enumerate(EMOTION_LABELS)
            },
        }

    @staticmethod
    def _generate_interpretation(
        emotion: str,
        confidence: float,
        gate_weights: dict[str, float],
        entropy_mean: float,
    ) -> str:
        """Generate a human-readable interpretation string.

        Args:
            emotion: Predicted emotion.
            confidence: Prediction confidence.
            gate_weights: Text vs audio gate weights.
            entropy_mean: Mean attention entropy.

        Returns:
            Interpretation string.
        """
        parts = []

        # Confidence assessment
        if confidence > 0.8:
            parts.append(f"High confidence ({confidence:.0%}) prediction of '{emotion}'.")
        elif confidence > 0.5:
            parts.append(f"Moderate confidence ({confidence:.0%}) prediction of '{emotion}'.")
        else:
            parts.append(f"Low confidence ({confidence:.0%}) prediction of '{emotion}'.")

        # Gate weight interpretation
        text_w = gate_weights["text"]
        audio_w = gate_weights["audio"]
        if text_w > audio_w + 0.15:
            parts.append("Model relied more on text content than vocal cues.")
        elif audio_w > text_w + 0.15:
            parts.append("Model relied more on vocal cues than text content.")
        else:
            parts.append("Model used both text and vocal cues roughly equally.")

        # Entropy interpretation
        if entropy_mean > 2.0:
            parts.append("Text is linguistically ambiguous (high attention entropy).")
        elif entropy_mean < 1.0:
            parts.append("Text is linguistically clear (low attention entropy).")

        return " ".join(parts)


# ---------------------------------------------------------------------------
# Standalone inference function (no class instantiation needed)
# ---------------------------------------------------------------------------

def run_inference(
    audio_path: str | Path,
    checkpoint_path: str | Path,
    config_path: str | Path = "configs/default.yaml",
    device: str = "auto",
) -> dict:
    """Convenience function for one-shot inference.

    Args:
        audio_path: Path to audio file.
        checkpoint_path: Path to MERA checkpoint.
        config_path: Path to config YAML.
        device: Device string.

    Returns:
        Inference result dict.
    """
    pipeline = MERAInference(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=device,
    )
    return pipeline.predict(audio_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MERA Single-File Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_inference.py --audio speech.wav
  python scripts/run_inference.py --audio speech.wav --checkpoint checkpoints/end_to_end/fold_1/best.pt
  python scripts/run_inference.py --audio speech.wav --output result.json --device cpu
        """,
    )
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to MERA checkpoint (default: checkpoints/end_to_end/fold_1/best.pt)",
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu", "cuda"])
    parser.add_argument("--output", type=str, default=None, help="Save result as JSON to this path")
    args = parser.parse_args()

    checkpoint = args.checkpoint or str(
        PROJECT_ROOT / "checkpoints" / "end_to_end" / "fold_1" / "best.pt"
    )

    pipeline = MERAInference(
        checkpoint_path=checkpoint,
        config_path=args.config,
        device=args.device,
    )

    result = pipeline.predict(args.audio)

    # Pretty-print result
    print("\n" + "=" * 60)
    print("MERA Emotion Recognition Result")
    print("=" * 60)
    print(f"  Audio:       {args.audio}")
    print(f"  Transcript:  \"{result['transcript']}\"")
    print(f"  Emotion:     {result['emotion'].upper()}")
    print(f"  Confidence:  {result['confidence']:.1%}")
    print()
    print("  Probabilities:")
    for label, prob in result["probabilities"].items():
        bar = "#" * int(prob * 40)
        print(f"    {label:>8s}: {prob:.3f}  {bar}")
    print()
    print(f"  Gate weights: text={result['gate_weights']['text']:.3f}, "
          f"audio={result['gate_weights']['audio']:.3f}")
    print(f"  Attention entropy (mean): {result['attention_entropy_mean']:.3f}")
    print()
    print(f"  Interpretation: {result['interpretation']}")
    print()
    print(f"  Text-only prediction:  {result['text_prediction']}")
    print(f"  Audio-only prediction: {result['audio_prediction']}")
    print("=" * 60)

    # Save JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResult saved to {output_path}")


if __name__ == "__main__":
    main()
