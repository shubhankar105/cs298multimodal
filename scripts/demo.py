#!/usr/bin/env python3
"""Gradio-based interactive demo for MERA emotion recognition.

Features:
1. Upload an audio file or record live audio.
2. Run full MERA inference pipeline.
3. Display an interpretability dashboard:
   - Emotion prediction with confidence bar
   - Probability distribution bar chart across all 4 emotions
   - Gate weight visualization (text vs audio stacked bar)
   - Transcript text
   - Prosodic contour plots (F0, energy, spectral centroid)
   - Attention entropy heatmap across DeBERTa's 12 layers
   - Spectrogram display

Runs entirely locally on Apple M5 Pro with no external API dependencies.

Usage::

    python scripts/demo.py
    python scripts/demo.py --checkpoint checkpoints/end_to_end/fold_1/best.pt
    python scripts/demo.py --share  # Create a public Gradio link
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

EMOTION_LABELS = ["angry", "happy", "sad", "neutral"]
EMOTION_COLORS = {
    "angry": "#e74c3c",
    "happy": "#f39c12",
    "sad": "#3498db",
    "neutral": "#95a5a6",
}


def create_demo(checkpoint_path: str, config_path: str = "configs/default.yaml", device: str = "auto"):
    """Build and return the Gradio demo interface.

    Args:
        checkpoint_path: Path to trained MERA checkpoint.
        config_path: Path to YAML config.
        device: Torch device string.

    Returns:
        A ``gr.Blocks`` interface ready to ``.launch()``.
    """
    try:
        import gradio as gr
    except ImportError:
        logger.error(
            "Gradio is required for the demo. Install with: pip install gradio"
        )
        sys.exit(1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        logger.error(
            "Matplotlib is required for the demo. Install with: pip install matplotlib"
        )
        sys.exit(1)

    # ---- Load inference pipeline ----
    from scripts.run_inference import MERAInference
    pipeline = MERAInference(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=device,
    )

    # ----------------------------------------------------------------
    # Plotting functions
    # ----------------------------------------------------------------

    def plot_probabilities(probabilities: dict) -> plt.Figure:
        """Bar chart of emotion probabilities."""
        fig, ax = plt.subplots(figsize=(6, 3))
        labels = list(probabilities.keys())
        values = list(probabilities.values())
        colors = [EMOTION_COLORS.get(l, "#333") for l in labels]
        bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.6)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.set_title("Emotion Probabilities")
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=10)
        fig.tight_layout()
        return fig

    def plot_gate_weights(gate_weights: dict) -> plt.Figure:
        """Horizontal stacked bar showing text vs audio contribution."""
        fig, ax = plt.subplots(figsize=(6, 1.5))
        text_w = gate_weights["text"]
        audio_w = gate_weights["audio"]
        ax.barh(["Modality\nWeight"], [text_w], color="#2ecc71", label=f"Text ({text_w:.1%})", height=0.5)
        ax.barh(["Modality\nWeight"], [audio_w], left=[text_w], color="#e67e22",
                label=f"Audio ({audio_w:.1%})", height=0.5)
        ax.set_xlim(0, 1)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title("Gate Weights: Text vs Audio Reliance")
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        fig.tight_layout()
        return fig

    def plot_attention_entropy(entropy: list) -> plt.Figure:
        """Heatmap of attention entropy across DeBERTa's 12 layers."""
        fig, ax = plt.subplots(figsize=(6, 2))
        entropy_arr = np.array(entropy).reshape(1, -1)
        im = ax.imshow(entropy_arr, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_yticks([])
        ax.set_xticks(range(len(entropy)))
        ax.set_xticklabels([f"L{i+1}" for i in range(len(entropy))], fontsize=8)
        ax.set_xlabel("DeBERTa Layer")
        ax.set_title(f"Attention Entropy (mean = {np.mean(entropy):.3f})")
        fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02, shrink=0.8)
        fig.tight_layout()
        return fig

    def plot_prosodic_contours(audio_path: str) -> plt.Figure:
        """Plot F0, energy, and spectral centroid from the audio."""
        fig, axes = plt.subplots(3, 1, figsize=(6, 5), sharex=True)

        try:
            from src.data.audio_utils import load_and_preprocess
            from src.features.prosodic import extract_prosodic_contours

            audio = load_and_preprocess(audio_path)
            contours = extract_prosodic_contours(audio)
            # contours shape: (10, T) — channels: F0, LogEnergy, SpectralCentroid, ...
            time_axis = np.arange(contours.shape[1]) * 0.01  # 10ms resolution

            # F0
            axes[0].plot(time_axis, contours[0], color="#e74c3c", linewidth=0.8)
            axes[0].set_ylabel("F0 (z-norm)")
            axes[0].set_title("Prosodic Contours")

            # Log Energy
            axes[1].plot(time_axis, contours[1], color="#2ecc71", linewidth=0.8)
            axes[1].set_ylabel("Energy (z-norm)")

            # Spectral Centroid
            axes[2].plot(time_axis, contours[2], color="#3498db", linewidth=0.8)
            axes[2].set_ylabel("Spec. Centroid (z-norm)")
            axes[2].set_xlabel("Time (s)")

        except Exception as e:
            for ax in axes:
                ax.text(0.5, 0.5, f"Could not extract contours:\n{e}",
                        ha="center", va="center", transform=ax.transAxes)

        fig.tight_layout()
        return fig

    def plot_spectrogram(audio_path: str) -> plt.Figure:
        """Display the log-mel spectrogram."""
        fig, ax = plt.subplots(figsize=(6, 3))

        try:
            from src.data.audio_utils import load_and_preprocess
            from src.features.spectrogram import extract_log_mel

            audio = load_and_preprocess(audio_path)
            spec = extract_log_mel(audio)
            # spec shape: (128, T), values in [0, 1]

            im = ax.imshow(spec, aspect="auto", origin="lower", cmap="magma",
                          interpolation="nearest")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Mel Bin")
            ax.set_title("Log-Mel Spectrogram")
            fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02, shrink=0.8)
        except Exception as e:
            ax.text(0.5, 0.5, f"Could not extract spectrogram:\n{e}",
                    ha="center", va="center", transform=ax.transAxes)

        fig.tight_layout()
        return fig

    # ----------------------------------------------------------------
    # Main inference callback
    # ----------------------------------------------------------------

    def process_audio(audio_input):
        """Run MERA inference and generate all visualizations."""
        if audio_input is None:
            return (
                "No audio provided.",
                None, None, None, None, None,
                "Please upload or record an audio file.",
            )

        # Gradio audio input can be (sample_rate, numpy_array) or filepath
        if isinstance(audio_input, tuple):
            # Recorded audio: save to temp file
            import tempfile
            import soundfile as sf
            sr, audio_data = audio_input
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)  # mono
            # Normalise to float32 [-1, 1]
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
                if np.abs(audio_data).max() > 1.0:
                    audio_data = audio_data / np.abs(audio_data).max()
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, audio_data, sr)
            audio_path = tmp.name
        else:
            audio_path = audio_input

        try:
            result = pipeline.predict(audio_path)
        except Exception as e:
            return (
                f"Inference error: {e}",
                None, None, None, None, None,
                f"Error: {e}",
            )

        # Build summary text
        summary = (
            f"**Predicted Emotion: {result['emotion'].upper()}** "
            f"(confidence: {result['confidence']:.1%})\n\n"
            f"**Transcript:** \"{result['transcript']}\"\n\n"
            f"**Gate Weights:** Text={result['gate_weights']['text']:.3f}, "
            f"Audio={result['gate_weights']['audio']:.3f}\n\n"
            f"**Attention Entropy (mean):** {result['attention_entropy_mean']:.3f}\n\n"
            f"**Interpretation:** {result['interpretation']}\n\n"
            f"---\n"
            f"Text-only prediction: {result['text_prediction']}  |  "
            f"Audio-only prediction: {result['audio_prediction']}"
        )

        # Generate plots
        prob_fig = plot_probabilities(result["probabilities"])
        gate_fig = plot_gate_weights(result["gate_weights"])
        entropy_fig = plot_attention_entropy(result["attention_entropy"])
        prosody_fig = plot_prosodic_contours(audio_path)
        spec_fig = plot_spectrogram(audio_path)

        return (
            summary,
            prob_fig,
            gate_fig,
            entropy_fig,
            prosody_fig,
            spec_fig,
            result["interpretation"],
        )

    # ----------------------------------------------------------------
    # Gradio interface layout
    # ----------------------------------------------------------------

    with gr.Blocks(
        title="MERA: Multimodal Emotion Recognition",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # MERA: Multimodal Emotion Recognition Architecture
            ### Cross-Modal Gated Attention Fusion for Speech Emotion Recognition

            Upload an audio file or record your voice to analyse the emotional content.
            The system uses both **text** (via DeBERTa) and **audio** (via CNN-BiLSTM +
            Prosodic TCN + HuBERT) with a novel **cross-modal gated attention fusion**.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Upload or Record Audio",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                submit_btn = gr.Button("Analyse Emotion", variant="primary", size="lg")

            with gr.Column(scale=2):
                summary_output = gr.Markdown(label="Result Summary")
                interpretation_output = gr.Textbox(
                    label="Interpretation",
                    interactive=False,
                    lines=2,
                )

        gr.Markdown("---\n### Interpretability Dashboard")

        with gr.Row():
            prob_plot = gr.Plot(label="Emotion Probabilities")
            gate_plot = gr.Plot(label="Gate Weights")

        with gr.Row():
            entropy_plot = gr.Plot(label="Attention Entropy Heatmap")

        with gr.Row():
            prosody_plot = gr.Plot(label="Prosodic Contours")
            spec_plot = gr.Plot(label="Spectrogram")

        # Connect
        submit_btn.click(
            fn=process_audio,
            inputs=[audio_input],
            outputs=[
                summary_output,
                prob_plot,
                gate_plot,
                entropy_plot,
                prosody_plot,
                spec_plot,
                interpretation_output,
            ],
        )

        gr.Markdown(
            """
            ---
            *MERA (Multimodal Emotion Recognition Architecture) — CS 298 Master's Thesis*

            **Novel contributions:** Prosodic contour TCN, attention entropy as
            linguistic confidence signal, modality dropout consistency training,
            three-stream paralinguistic pipeline with SE weighting.
            """
        )

    return demo


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MERA Interactive Demo")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to MERA checkpoint",
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    checkpoint = args.checkpoint or str(
        PROJECT_ROOT / "checkpoints" / "end_to_end" / "fold_1" / "best.pt"
    )

    demo = create_demo(
        checkpoint_path=checkpoint,
        config_path=args.config,
        device=args.device,
    )
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
