"""Attention entropy computation for DeBERTa layers.

**Novel contribution**: Measures per-layer uncertainty in DeBERTa's
self-attention distributions.  When the model is "confused" about which
tokens matter (e.g. sarcasm, mixed signals), attention entropy is high.
The fusion module uses this signal to decide whether to trust the text
pipeline or defer to audio cues.

High entropy  → attention spread out → text is ambiguous → rely on audio.
Low entropy   → attention focused    → text is confident → rely on text.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionEntropyModule(nn.Module):
    """Compute per-layer attention entropy from attention weight tensors.

    Given the raw attention weight tuple from a transformer encoder, returns
    a ``(batch, num_layers)`` tensor of average Shannon entropy values.
    """

    def __init__(self, eps: float = 1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, attentions: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Compute mean attention entropy per layer.

        Args:
            attentions: Tuple of ``(batch, heads, seq_len, seq_len)`` tensors,
                one per transformer layer.

        Returns:
            ``(batch, num_layers)`` tensor of average entropy values.
        """
        layer_entropies = []
        for layer_attn in attentions:
            # layer_attn: (B, H, S, S) — already softmaxed attention weights
            attn = layer_attn + self.eps

            # Shannon entropy: H = -Σ p·log(p) along key dimension
            entropy = -torch.sum(attn * torch.log(attn), dim=-1)  # (B, H, S)

            # Average over heads and sequence positions → scalar per sample
            avg_entropy = entropy.mean(dim=(1, 2))  # (B,)
            layer_entropies.append(avg_entropy)

        return torch.stack(layer_entropies, dim=1)  # (B, L)


def compute_attention_entropy(
    attentions: tuple[torch.Tensor, ...],
    eps: float = 1e-10,
) -> torch.Tensor:
    """Functional interface for attention entropy computation.

    Args:
        attentions: Tuple of ``(batch, heads, seq_len, seq_len)`` tensors.
        eps: Small constant for numerical stability.

    Returns:
        ``(batch, num_layers)`` tensor.
    """
    return AttentionEntropyModule(eps=eps)(attentions)
