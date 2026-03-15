"""MERA: Complete end-to-end multimodal emotion recognition model.

Combines Pipeline A (text), Pipeline B (audio), and the novel cross-modal
gated attention fusion module into a single ``nn.Module``.

Training modes
--------------
1. ``PIPELINE_A_ONLY``  — train text pipeline in isolation.
2. ``PIPELINE_B_ONLY``  — train audio pipeline in isolation.
3. ``FUSION_ONLY``      — freeze both pipelines, train fusion + classifier.
4. ``END_TO_END``       — fine-tune last 2 DeBERTa layers + all of Pipeline B
                          + fusion + final classifier.

Memory management (24 GB M5 Pro)
    FUSION_ONLY  ≈ 4–6 GB  (pipelines frozen, no gradients stored)
    END_TO_END   ≈ 16–18 GB (gradient checkpointing + float32)
"""

from __future__ import annotations

from enum import Enum

import torch
import torch.nn as nn

from src.models.pipeline_a.text_encoder import TextEmotionEncoder
from src.models.pipeline_b.audio_emotion_head import AudioEmotionHead
from src.models.fusion.cross_modal_attention import CrossModalAttention
from src.models.fusion.gated_fusion import GatedFusion


class TrainingMode(Enum):
    PIPELINE_A_ONLY = "pipeline_a"
    PIPELINE_B_ONLY = "pipeline_b"
    FUSION_ONLY = "fusion"
    END_TO_END = "end_to_end"


class MERAModel(nn.Module):
    """Multimodal Emotion Recognition Architecture."""

    def __init__(
        self,
        num_classes: int = 4,
        text_model_name: str = "microsoft/deberta-v3-base",
        embed_dim: int = 256,
        dropout: float = 0.3,
        modality_dropout_prob: float = 0.3,
        # Pipeline B passthrough
        n_mels: int = 128,
        cnn_channels: list[int] | None = None,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        tcn_channels: int = 64,
        tcn_blocks: int = 6,
        tcn_kernel_size: int = 5,
        prosodic_output_dim: int = 128,
        hubert_num_layers: int = 25,
        hubert_hidden_dim: int = 1024,
        se_reduction: int = 4,
        cross_attention_heads: int = 4,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.modality_dropout_prob = modality_dropout_prob

        # ---- Pipeline A: Linguistic ----
        self.pipeline_a = TextEmotionEncoder(
            model_name=text_model_name,
            num_classes=num_classes,
            hidden_dim=embed_dim,
            dropout=dropout,
        )

        # ---- Pipeline B: Paralinguistic ----
        self.pipeline_b = AudioEmotionHead(
            num_classes=num_classes,
            embed_dim=embed_dim,
            dropout=dropout,
            n_mels=n_mels,
            cnn_channels=cnn_channels,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            tcn_channels=tcn_channels,
            tcn_blocks=tcn_blocks,
            tcn_kernel_size=tcn_kernel_size,
            prosodic_output_dim=prosodic_output_dim,
            hubert_num_layers=hubert_num_layers,
            hubert_hidden_dim=hubert_hidden_dim,
            se_reduction=se_reduction,
        )

        # ---- Fusion ----
        self.cross_modal_attention = CrossModalAttention(
            dim=embed_dim, num_heads=cross_attention_heads,
        )
        self.gated_fusion = GatedFusion(
            repr_dim=embed_dim,
            entropy_dim=self.pipeline_a.num_layers,   # 12 for DeBERTa-base
            dropout=dropout,
        )

        # ---- Final classifier ----
        self.final_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        spectrogram: torch.Tensor,
        spec_mask: torch.Tensor | None,
        prosody: torch.Tensor,
        pros_mask: torch.Tensor | None,
        hubert: torch.Tensor,
        hub_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Full MERA forward pass.

        Returns a dict with:
        - ``final_logits``      ``(B, C)``  — main prediction
        - ``text_logits``       ``(B, C)``  — Pipeline A standalone
        - ``audio_logits``      ``(B, C)``  — Pipeline B standalone
        - ``gate_weights``      ``(B, 2)``  — fusion gate (interpretability)
        - ``attention_entropy`` ``(B, L)``  — linguistic confidence
        """
        outputs: dict[str, torch.Tensor] = {}

        # ---- Pipeline A: Text ----
        text_logits, text_repr, attention_entropy = self.pipeline_a(
            input_ids, attention_mask,
        )
        outputs["text_logits"] = text_logits
        outputs["attention_entropy"] = attention_entropy

        # ---- Pipeline B: Audio ----
        audio_logits, audio_repr = self.pipeline_b(
            spectrogram, spec_mask, prosody, pros_mask, hubert, hub_mask,
        )
        outputs["audio_logits"] = audio_logits

        # ---- Modality dropout (training only) ----
        if self.training and self.modality_dropout_prob > 0:
            rand_val = torch.rand(1).item()
            if rand_val < self.modality_dropout_prob / 2:
                # Drop text
                text_repr = torch.zeros_like(text_repr)
                attention_entropy = torch.zeros_like(attention_entropy)
            elif rand_val < self.modality_dropout_prob:
                # Drop audio
                audio_repr = torch.zeros_like(audio_repr)

        # ---- Cross-modal attention ----
        text_enhanced, audio_enhanced = self.cross_modal_attention(
            text_repr, audio_repr,
        )

        # ---- Gated fusion ----
        fused, gate_weights = self.gated_fusion(
            text_enhanced, audio_enhanced, attention_entropy,
        )
        outputs["gate_weights"] = gate_weights

        # ---- Final classification ----
        outputs["final_logits"] = self.final_classifier(fused)

        return outputs

    # ------------------------------------------------------------------
    # Training-mode configuration
    # ------------------------------------------------------------------

    def set_training_mode(self, mode: TrainingMode) -> None:
        """Configure which components are trainable.

        Args:
            mode: One of the four ``TrainingMode`` enum values.
        """
        if mode == TrainingMode.PIPELINE_A_ONLY:
            self._freeze_all()
            self._unfreeze(self.pipeline_a)

        elif mode == TrainingMode.PIPELINE_B_ONLY:
            self._freeze_all()
            self._unfreeze(self.pipeline_b)

        elif mode == TrainingMode.FUSION_ONLY:
            self._freeze_all()
            self._unfreeze(self.cross_modal_attention)
            self._unfreeze(self.gated_fusion)
            self._unfreeze(self.final_classifier)

        elif mode == TrainingMode.END_TO_END:
            self._freeze_all()
            # Unfreeze last 2 DeBERTa layers
            for layer in self.pipeline_a.deberta.encoder.layer[-2:]:
                self._unfreeze(layer)
            self._unfreeze(self.pipeline_a.classifier)
            self._unfreeze(self.pipeline_a.representation_proj)
            # Unfreeze all of Pipeline B
            self._unfreeze(self.pipeline_b)
            # Unfreeze fusion + final classifier
            self._unfreeze(self.cross_modal_attention)
            self._unfreeze(self.gated_fusion)
            self._unfreeze(self.final_classifier)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _freeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def _unfreeze(self, module: nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad = True
