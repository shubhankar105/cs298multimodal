"""Complete Pipeline B: Three audio streams + SE fusion + classification.

Combines:
1. CNN-BiLSTM (spectrogram)  → 256-d
2. Prosodic TCN (contours)   → 128-d  ← novel
3. HuBERT weighted head       → 256-d

Squeeze-and-Excitation fuses these into a single 256-d audio representation,
which is then classified *and* passed to the fusion module.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.pipeline_b.cnn_bilstm import CNNBiLSTM
from src.models.pipeline_b.prosodic_tcn import ProsodicTCN
from src.models.pipeline_b.hubert_head import HuBERTWeightedHead
from src.models.pipeline_b.squeeze_excitation import SubStreamFusion


class AudioEmotionHead(nn.Module):
    """Full Pipeline B with three streams, SE fusion, and classifier."""

    def __init__(
        self,
        num_classes: int = 4,
        embed_dim: int = 256,
        dropout: float = 0.3,
        # Stream 1 (CNN-BiLSTM) params
        n_mels: int = 128,
        cnn_channels: list[int] | None = None,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        # Stream 2 (Prosodic TCN) params
        prosodic_in_channels: int = 10,
        tcn_channels: int = 64,
        tcn_blocks: int = 6,
        tcn_kernel_size: int = 5,
        prosodic_output_dim: int = 128,
        # Stream 3 (HuBERT) params
        hubert_num_layers: int = 25,
        hubert_hidden_dim: int = 1024,
        # SE params
        se_reduction: int = 4,
    ):
        super().__init__()

        self.stream1 = CNNBiLSTM(
            n_mels=n_mels,
            cnn_channels=cnn_channels,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            output_dim=embed_dim,
            dropout=dropout,
        )

        self.stream2 = ProsodicTCN(
            in_channels=prosodic_in_channels,
            tcn_channels=tcn_channels,
            num_blocks=tcn_blocks,
            kernel_size=tcn_kernel_size,
            output_dim=prosodic_output_dim,
            dropout=dropout * 0.67,
        )

        self.stream3 = HuBERTWeightedHead(
            num_layers=hubert_num_layers,
            hidden_dim=hubert_hidden_dim,
            output_dim=embed_dim,
            dropout=dropout,
        )

        self.fusion = SubStreamFusion(
            stream1_dim=embed_dim,
            stream2_dim=prosodic_output_dim,
            stream3_dim=embed_dim,
            output_dim=embed_dim,
            reduction=se_reduction,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(
        self,
        spectrogram: torch.Tensor,
        spec_mask: torch.Tensor | None,
        prosody: torch.Tensor,
        pros_mask: torch.Tensor | None,
        hubert: torch.Tensor,
        hub_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            spectrogram: ``(B, 128, T_s)``
            spec_mask:   ``(B, T_s)``
            prosody:     ``(B, 10, T_p)``
            pros_mask:   ``(B, T_p)``
            hubert:      ``(B, 25, T_h, 1024)``
            hub_mask:    ``(B, T_h)``

        Returns:
            logits:    ``(B, num_classes)``
            embedding: ``(B, embed_dim)`` — for fusion module.
        """
        s1 = self.stream1(spectrogram, spec_mask)   # (B, 256)
        s2 = self.stream2(prosody, pros_mask)        # (B, 128)
        s3 = self.stream3(hubert, hub_mask)          # (B, 256)

        fused = self.fusion(s1, s2, s3)              # (B, 256)
        logits = self.classifier(fused)              # (B, C)

        return logits, fused
