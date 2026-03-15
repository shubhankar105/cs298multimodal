"""Stream 1: Spectro-temporal CNN-BiLSTM for spectrogram-based emotion recognition.

Architecture
------------
1. 4-layer CNN with batch-norm extracts local spectral patterns from
   the log-Mel spectrogram.
2. 2-layer BiLSTM captures temporal dynamics across the utterance.
3. Attention pooling over BiLSTM outputs produces a fixed-size embedding.

Shape flow::

    Input:  (batch, 128, T)  — log-Mel spectrogram
    CNN:    (batch, C, F', T) — F' = 128 / 2⁴ = 8
    Reshape:(batch, T, C·F')
    LSTM:   (batch, T, 2·H)
    Pool:   (batch, 2·H)
    Output: (batch, output_dim)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CNNBiLSTM(nn.Module):
    """Spectro-temporal CNN-BiLSTM with attention pooling."""

    def __init__(
        self,
        n_mels: int = 128,
        cnn_channels: list[int] | None = None,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        output_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        if cnn_channels is None:
            cnn_channels = [32, 64, 128, 128]

        # ---- CNN feature extractor ----
        layers: list[nn.Module] = []
        in_ch = 1
        for out_ch in cnn_channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1)),   # pool freq only, preserve time
                nn.Dropout2d(dropout * 0.5),
            ])
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        # freq dimension after pooling: 128 / 2^4 = 8
        cnn_freq_out = n_mels // (2 ** len(cnn_channels))
        lstm_input_dim = cnn_channels[-1] * cnn_freq_out

        # ---- BiLSTM ----
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # ---- Attention pooling ----
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.Tanh(),
            nn.Linear(lstm_hidden, 1),
        )

        # ---- Output projection ----
        self.projection = nn.Linear(lstm_hidden * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        spectrogram: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            spectrogram: ``(batch, 128, T)`` log-Mel spectrogram.
            mask: ``(batch, T)`` boolean mask (True = valid).

        Returns:
            ``(batch, output_dim)`` embedding.
        """
        # (B, 1, 128, T)
        x = spectrogram.unsqueeze(1)

        # CNN → (B, C, F', T)
        x = self.cnn(x)
        batch, channels, freq, time = x.shape

        # Reshape for LSTM → (B, T, C·F')
        x = x.permute(0, 3, 1, 2).contiguous().view(batch, time, channels * freq)

        # BiLSTM → (B, T, 2H)
        x, _ = self.lstm(x)

        # Attention scores → (B, T)
        attn_scores = self.attention(x).squeeze(-1)

        # Apply mask (padded positions → -inf)
        if mask is not None:
            m = mask[:, :time] if mask.shape[1] > time else mask
            if m.shape[1] < time:
                m = nn.functional.pad(m, (0, time - m.shape[1]), value=False)
            attn_scores = attn_scores.masked_fill(~m, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(1)  # (B, 1, T)
        pooled = torch.bmm(attn_weights, x).squeeze(1)                # (B, 2H)

        embedding = self.projection(self.dropout(pooled))              # (B, D)
        return embedding
