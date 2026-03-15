"""Tests for all MERA model architectures.

All tests use small random tensors on CPU so they run fast without
requiring large model downloads.  DeBERTa-dependent tests are isolated
and skip gracefully when transformers models are not cached.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

# Fixed small sizes for fast tests
B = 2       # batch
T = 50      # spectrogram / prosodic time frames
T_HUB = 25  # HuBERT time frames (lower frame rate)
N_MELS = 128
N_PROS = 10
D = 256     # embed_dim
NUM_CLASSES = 4


# ---------------------------------------------------------------------------
# Attention Entropy
# ---------------------------------------------------------------------------

class TestAttentionEntropy:
    """Tests for the attention entropy module."""

    def test_output_shape(self):
        from src.models.pipeline_a.attention_entropy import AttentionEntropyModule

        module = AttentionEntropyModule()
        num_layers = 12
        # Simulate softmaxed attention weights
        attentions = tuple(
            torch.softmax(torch.randn(B, 12, 20, 20), dim=-1)
            for _ in range(num_layers)
        )
        entropy = module(attentions)
        assert entropy.shape == (B, num_layers)

    def test_values_non_negative(self):
        from src.models.pipeline_a.attention_entropy import AttentionEntropyModule

        module = AttentionEntropyModule()
        attentions = tuple(
            torch.softmax(torch.randn(B, 4, 10, 10), dim=-1)
            for _ in range(6)
        )
        entropy = module(attentions)
        assert (entropy >= 0).all()

    def test_uniform_vs_peaked(self):
        """Uniform attention should have higher entropy than peaked."""
        from src.models.pipeline_a.attention_entropy import AttentionEntropyModule

        module = AttentionEntropyModule()
        seq_len = 10

        # Uniform attention
        uniform = torch.ones(1, 1, seq_len, seq_len) / seq_len
        # Peaked attention (one-hot)
        peaked = torch.zeros(1, 1, seq_len, seq_len)
        peaked[:, :, :, 0] = 1.0

        e_uniform = module((uniform,))
        e_peaked = module((peaked,))
        assert e_uniform.item() > e_peaked.item()


# ---------------------------------------------------------------------------
# CNN-BiLSTM (Stream 1)
# ---------------------------------------------------------------------------

class TestCNNBiLSTM:
    """Tests for the CNN-BiLSTM spectrogram stream."""

    def test_output_shape(self):
        from src.models.pipeline_b.cnn_bilstm import CNNBiLSTM

        model = CNNBiLSTM(n_mels=N_MELS, output_dim=D)
        spec = torch.randn(B, N_MELS, T)
        out = model(spec)
        assert out.shape == (B, D)

    def test_with_mask(self):
        from src.models.pipeline_b.cnn_bilstm import CNNBiLSTM

        model = CNNBiLSTM(n_mels=N_MELS, output_dim=D)
        spec = torch.randn(B, N_MELS, T)
        mask = torch.ones(B, T, dtype=torch.bool)
        mask[0, T // 2:] = False  # Mask second half for first sample
        out = model(spec, mask=mask)
        assert out.shape == (B, D)

    def test_gradient_flow(self):
        from src.models.pipeline_b.cnn_bilstm import CNNBiLSTM

        model = CNNBiLSTM(n_mels=N_MELS, output_dim=D)
        spec = torch.randn(B, N_MELS, T, requires_grad=True)
        out = model(spec)
        loss = out.sum()
        loss.backward()
        assert spec.grad is not None
        assert spec.grad.abs().sum() > 0

    def test_different_time_lengths(self):
        from src.models.pipeline_b.cnn_bilstm import CNNBiLSTM

        model = CNNBiLSTM(n_mels=N_MELS, output_dim=D)
        for t in [20, 50, 100, 200]:
            out = model(torch.randn(1, N_MELS, t))
            assert out.shape == (1, D)


# ---------------------------------------------------------------------------
# Prosodic TCN (Stream 2 — NOVEL)
# ---------------------------------------------------------------------------

class TestProsodicTCN:
    """Tests for the novel Prosodic TCN."""

    def test_output_shape(self):
        from src.models.pipeline_b.prosodic_tcn import ProsodicTCN

        model = ProsodicTCN(in_channels=N_PROS, output_dim=128)
        prosody = torch.randn(B, N_PROS, T)
        out = model(prosody)
        assert out.shape == (B, 128)

    def test_with_mask(self):
        from src.models.pipeline_b.prosodic_tcn import ProsodicTCN

        model = ProsodicTCN(in_channels=N_PROS, output_dim=128)
        prosody = torch.randn(B, N_PROS, T)
        mask = torch.ones(B, T, dtype=torch.bool)
        mask[1, T // 2:] = False
        out = model(prosody, mask=mask)
        assert out.shape == (B, 128)

    def test_temporal_block(self):
        from src.models.pipeline_b.prosodic_tcn import TemporalBlock

        block = TemporalBlock(64, 64, kernel_size=5, dilation=4)
        x = torch.randn(B, 64, T)
        out = block(x)
        assert out.shape == (B, 64, T), "TemporalBlock must preserve time dimension"

    def test_temporal_block_channel_change(self):
        from src.models.pipeline_b.prosodic_tcn import TemporalBlock

        block = TemporalBlock(10, 64, kernel_size=5, dilation=1)
        x = torch.randn(B, 10, T)
        out = block(x)
        assert out.shape == (B, 64, T)

    def test_channel_attention(self):
        from src.models.pipeline_b.prosodic_tcn import ChannelAttention

        ca = ChannelAttention(N_PROS, reduction=2)
        x = torch.randn(B, N_PROS, T)
        out = ca(x)
        assert out.shape == x.shape

    def test_gradient_flow(self):
        from src.models.pipeline_b.prosodic_tcn import ProsodicTCN

        model = ProsodicTCN(in_channels=N_PROS, output_dim=128)
        prosody = torch.randn(B, N_PROS, T, requires_grad=True)
        out = model(prosody)
        loss = out.sum()
        loss.backward()
        assert prosody.grad is not None

    def test_dilation_pattern(self):
        """Verify TCN blocks have the expected exponential dilation pattern."""
        from src.models.pipeline_b.prosodic_tcn import ProsodicTCN

        model = ProsodicTCN(num_blocks=6)
        dilations = [block.conv1.dilation[0] for block in model.blocks]
        assert dilations == [1, 2, 4, 8, 16, 32]


# ---------------------------------------------------------------------------
# HuBERT Weighted Head (Stream 3)
# ---------------------------------------------------------------------------

class TestHuBERTHead:
    """Tests for the HuBERT weighted-sum head."""

    def test_output_shape(self):
        from src.models.pipeline_b.hubert_head import HuBERTWeightedHead

        model = HuBERTWeightedHead(num_layers=25, hidden_dim=1024, output_dim=D)
        hubert = torch.randn(B, 25, T_HUB, 1024)
        out = model(hubert)
        assert out.shape == (B, D)

    def test_with_mask(self):
        from src.models.pipeline_b.hubert_head import HuBERTWeightedHead

        model = HuBERTWeightedHead(num_layers=25, hidden_dim=1024, output_dim=D)
        hubert = torch.randn(B, 25, T_HUB, 1024)
        mask = torch.ones(B, T_HUB, dtype=torch.bool)
        mask[0, 15:] = False
        out = model(hubert, mask=mask)
        assert out.shape == (B, D)

    def test_layer_weights_sum_to_one(self):
        from src.models.pipeline_b.hubert_head import HuBERTWeightedHead

        model = HuBERTWeightedHead()
        weights = torch.softmax(model.layer_weights, dim=0)
        assert abs(weights.sum().item() - 1.0) < 1e-5

    def test_gradient_flow(self):
        from src.models.pipeline_b.hubert_head import HuBERTWeightedHead

        model = HuBERTWeightedHead(num_layers=25, hidden_dim=1024, output_dim=D)
        hubert = torch.randn(B, 25, T_HUB, 1024, requires_grad=True)
        out = model(hubert)
        loss = out.sum()
        loss.backward()
        assert hubert.grad is not None
        # Layer weights should also get gradients
        assert model.layer_weights.grad is not None


# ---------------------------------------------------------------------------
# SE Sub-Stream Fusion
# ---------------------------------------------------------------------------

class TestSubStreamFusion:
    """Tests for the Squeeze-and-Excitation sub-stream fusion."""

    def test_output_shape(self):
        from src.models.pipeline_b.squeeze_excitation import SubStreamFusion

        model = SubStreamFusion(stream1_dim=256, stream2_dim=128, stream3_dim=256, output_dim=D)
        s1 = torch.randn(B, 256)
        s2 = torch.randn(B, 128)
        s3 = torch.randn(B, 256)
        out = model(s1, s2, s3)
        assert out.shape == (B, D)


# ---------------------------------------------------------------------------
# Audio Emotion Head (full Pipeline B)
# ---------------------------------------------------------------------------

class TestAudioEmotionHead:
    """Tests for the complete Pipeline B."""

    def test_output_shapes(self):
        from src.models.pipeline_b.audio_emotion_head import AudioEmotionHead

        model = AudioEmotionHead(num_classes=NUM_CLASSES, embed_dim=D)
        spec = torch.randn(B, N_MELS, T)
        pros = torch.randn(B, N_PROS, T)
        hub = torch.randn(B, 25, T_HUB, 1024)

        logits, embedding = model(spec, None, pros, None, hub, None)
        assert logits.shape == (B, NUM_CLASSES)
        assert embedding.shape == (B, D)

    def test_with_masks(self):
        from src.models.pipeline_b.audio_emotion_head import AudioEmotionHead

        model = AudioEmotionHead(num_classes=NUM_CLASSES, embed_dim=D)
        spec = torch.randn(B, N_MELS, T)
        pros = torch.randn(B, N_PROS, T)
        hub = torch.randn(B, 25, T_HUB, 1024)
        spec_mask = torch.ones(B, T, dtype=torch.bool)
        pros_mask = torch.ones(B, T, dtype=torch.bool)
        hub_mask = torch.ones(B, T_HUB, dtype=torch.bool)

        logits, embedding = model(spec, spec_mask, pros, pros_mask, hub, hub_mask)
        assert logits.shape == (B, NUM_CLASSES)


# ---------------------------------------------------------------------------
# Cross-Modal Attention
# ---------------------------------------------------------------------------

class TestCrossModalAttention:
    """Tests for the bidirectional cross-modal attention."""

    def test_output_shapes(self):
        from src.models.fusion.cross_modal_attention import CrossModalAttention

        model = CrossModalAttention(dim=D, num_heads=4)
        t = torch.randn(B, D)
        a = torch.randn(B, D)
        t_enh, a_enh = model(t, a)
        assert t_enh.shape == (B, D)
        assert a_enh.shape == (B, D)

    def test_residual_connection(self):
        """Output should differ from input (cross-attention adds information)."""
        from src.models.fusion.cross_modal_attention import CrossModalAttention

        model = CrossModalAttention(dim=D)
        t = torch.randn(B, D)
        a = torch.randn(B, D)
        t_enh, a_enh = model(t, a)

        # Should not be identical to input (attention adds cross-modal info)
        assert not torch.allclose(t_enh, t, atol=1e-4)


# ---------------------------------------------------------------------------
# Gated Fusion
# ---------------------------------------------------------------------------

class TestGatedFusion:
    """Tests for the entropy-informed gated fusion."""

    def test_output_shapes(self):
        from src.models.fusion.gated_fusion import GatedFusion

        model = GatedFusion(repr_dim=D, entropy_dim=12)
        t = torch.randn(B, D)
        a = torch.randn(B, D)
        ent = torch.randn(B, 12)
        fused, gate_weights = model(t, a, ent)

        assert fused.shape == (B, D)
        assert gate_weights.shape == (B, 2)

    def test_gate_weights_sum_to_one(self):
        from src.models.fusion.gated_fusion import GatedFusion

        model = GatedFusion(repr_dim=D, entropy_dim=12)
        t = torch.randn(B, D)
        a = torch.randn(B, D)
        ent = torch.randn(B, 12)
        _, gate_weights = model(t, a, ent)

        sums = gate_weights.sum(dim=1)
        assert torch.allclose(sums, torch.ones(B), atol=1e-4)

    def test_gate_weights_non_negative(self):
        from src.models.fusion.gated_fusion import GatedFusion

        model = GatedFusion(repr_dim=D, entropy_dim=12)
        _, gate_weights = model(torch.randn(B, D), torch.randn(B, D), torch.randn(B, 12))
        assert (gate_weights >= 0).all()


# ---------------------------------------------------------------------------
# Full MERA Model (without DeBERTa download — mock Pipeline A)
# ---------------------------------------------------------------------------

class TestMERAModelMocked:
    """Tests for the full MERA model using a lightweight mock Pipeline A.

    This avoids downloading the 400 MB DeBERTa checkpoint while still
    testing the complete forward pass, training modes, and gradient flow.
    """

    class MockTextEncoder(nn.Module):
        """Lightweight stand-in for TextEmotionEncoder."""

        def __init__(self, hidden_dim=256, num_classes=4, num_layers=12):
            super().__init__()
            self.num_layers = num_layers
            self.hidden_size = 768
            self.num_heads = 12
            self.classifier = nn.Linear(hidden_dim, num_classes)
            self.representation_proj = nn.Linear(32, hidden_dim)
            self.entropy_module = nn.Identity()
            self._proj = nn.Linear(32, hidden_dim)
            self._cls = nn.Linear(hidden_dim, num_classes)

            # Mock deberta with a fake encoder.layer for training mode tests
            self.deberta = nn.Module()
            self.deberta.encoder = nn.Module()
            self.deberta.encoder.layer = nn.ModuleList([
                nn.Linear(10, 10) for _ in range(num_layers)
            ])

        def forward(self, input_ids, attention_mask):
            B = input_ids.shape[0]
            # Fake representations
            repr_ = self._proj(torch.randn(B, 32, device=input_ids.device))
            logits = self._cls(repr_)
            entropy = torch.randn(B, self.num_layers, device=input_ids.device).abs()
            return logits, repr_, entropy

    def _build_model(self):
        """Build a MERA model with mocked Pipeline A."""
        from src.models.fusion.mera_model import MERAModel

        model = MERAModel.__new__(MERAModel)
        nn.Module.__init__(model)

        model.num_classes = NUM_CLASSES
        model.modality_dropout_prob = 0.3

        # Mock Pipeline A
        model.pipeline_a = self.MockTextEncoder(
            hidden_dim=D, num_classes=NUM_CLASSES,
        )

        # Real Pipeline B (lightweight)
        from src.models.pipeline_b.audio_emotion_head import AudioEmotionHead
        model.pipeline_b = AudioEmotionHead(
            num_classes=NUM_CLASSES, embed_dim=D,
        )

        # Real fusion
        from src.models.fusion.cross_modal_attention import CrossModalAttention
        from src.models.fusion.gated_fusion import GatedFusion

        model.cross_modal_attention = CrossModalAttention(dim=D)
        model.gated_fusion = GatedFusion(repr_dim=D, entropy_dim=12)
        model.final_classifier = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(D // 2, NUM_CLASSES),
        )

        # Bind methods from MERAModel
        from src.models.fusion.mera_model import MERAModel as _M
        model.forward = _M.forward.__get__(model, type(model))
        model.set_training_mode = _M.set_training_mode.__get__(model, type(model))
        model._freeze_all = _M._freeze_all.__get__(model, type(model))
        model._unfreeze = _M._unfreeze.__get__(model, type(model))

        return model

    def _make_inputs(self):
        return {
            "input_ids": torch.randint(0, 100, (B, 20)),
            "attention_mask": torch.ones(B, 20, dtype=torch.long),
            "spectrogram": torch.randn(B, N_MELS, T),
            "spec_mask": torch.ones(B, T, dtype=torch.bool),
            "prosody": torch.randn(B, N_PROS, T),
            "pros_mask": torch.ones(B, T, dtype=torch.bool),
            "hubert": torch.randn(B, 25, T_HUB, 1024),
            "hub_mask": torch.ones(B, T_HUB, dtype=torch.bool),
        }

    def test_forward_pass(self):
        model = self._build_model()
        model.eval()
        inputs = self._make_inputs()

        with torch.no_grad():
            outputs = model(**inputs)

        assert "final_logits" in outputs
        assert "text_logits" in outputs
        assert "audio_logits" in outputs
        assert "gate_weights" in outputs
        assert "attention_entropy" in outputs

        assert outputs["final_logits"].shape == (B, NUM_CLASSES)
        assert outputs["text_logits"].shape == (B, NUM_CLASSES)
        assert outputs["audio_logits"].shape == (B, NUM_CLASSES)
        assert outputs["gate_weights"].shape == (B, 2)

    def test_forward_train_mode(self):
        """Forward pass in training mode (modality dropout active)."""
        model = self._build_model()
        model.train()
        inputs = self._make_inputs()
        outputs = model(**inputs)
        assert outputs["final_logits"].shape == (B, NUM_CLASSES)

    def test_training_mode_pipeline_a_only(self):
        from src.models.fusion.mera_model import TrainingMode

        model = self._build_model()
        model.set_training_mode(TrainingMode.PIPELINE_A_ONLY)

        # Pipeline A params should be trainable
        for p in model.pipeline_a.parameters():
            assert p.requires_grad

        # Pipeline B should be frozen
        for p in model.pipeline_b.parameters():
            assert not p.requires_grad

        # Fusion should be frozen
        for p in model.cross_modal_attention.parameters():
            assert not p.requires_grad

    def test_training_mode_pipeline_b_only(self):
        from src.models.fusion.mera_model import TrainingMode

        model = self._build_model()
        model.set_training_mode(TrainingMode.PIPELINE_B_ONLY)

        for p in model.pipeline_b.parameters():
            assert p.requires_grad

        for p in model.pipeline_a.parameters():
            assert not p.requires_grad

    def test_training_mode_fusion_only(self):
        from src.models.fusion.mera_model import TrainingMode

        model = self._build_model()
        model.set_training_mode(TrainingMode.FUSION_ONLY)

        # Fusion + classifier should be trainable
        for p in model.cross_modal_attention.parameters():
            assert p.requires_grad
        for p in model.gated_fusion.parameters():
            assert p.requires_grad
        for p in model.final_classifier.parameters():
            assert p.requires_grad

        # Pipelines should be frozen
        for p in model.pipeline_a.parameters():
            assert not p.requires_grad
        for p in model.pipeline_b.parameters():
            assert not p.requires_grad

    def test_training_mode_end_to_end(self):
        from src.models.fusion.mera_model import TrainingMode

        model = self._build_model()
        model.set_training_mode(TrainingMode.END_TO_END)

        # Last 2 DeBERTa layers should be unfrozen
        for layer in model.pipeline_a.deberta.encoder.layer[-2:]:
            for p in layer.parameters():
                assert p.requires_grad

        # First 10 layers should be frozen
        for layer in model.pipeline_a.deberta.encoder.layer[:-2]:
            for p in layer.parameters():
                assert not p.requires_grad

        # Pipeline B should be unfrozen
        for p in model.pipeline_b.parameters():
            assert p.requires_grad

        # Fusion should be unfrozen
        for p in model.cross_modal_attention.parameters():
            assert p.requires_grad

    def test_gradient_flow_fusion_only(self):
        from src.models.fusion.mera_model import TrainingMode

        model = self._build_model()
        model.set_training_mode(TrainingMode.FUSION_ONLY)
        model.train()

        inputs = self._make_inputs()
        outputs = model(**inputs)
        loss = outputs["final_logits"].sum()
        loss.backward()

        # Fusion params should have gradients
        for name, p in model.gated_fusion.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for gated_fusion.{name}"

    def test_all_training_modes_forward(self):
        """All four training modes should produce valid outputs."""
        from src.models.fusion.mera_model import TrainingMode

        model = self._build_model()
        inputs = self._make_inputs()

        for mode in TrainingMode:
            model.set_training_mode(mode)
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
            assert outputs["final_logits"].shape == (B, NUM_CLASSES), \
                f"Failed for mode {mode}"


# ---------------------------------------------------------------------------
# Integration: Pipeline B end-to-end gradient test
# ---------------------------------------------------------------------------

class TestPipelineBGradient:
    """End-to-end gradient flow test for Pipeline B."""

    def test_full_pipeline_b_backward(self):
        from src.models.pipeline_b.audio_emotion_head import AudioEmotionHead

        model = AudioEmotionHead(num_classes=NUM_CLASSES, embed_dim=D)
        model.train()

        spec = torch.randn(B, N_MELS, T, requires_grad=True)
        pros = torch.randn(B, N_PROS, T, requires_grad=True)
        hub = torch.randn(B, 25, T_HUB, 1024, requires_grad=True)

        logits, emb = model(spec, None, pros, None, hub, None)
        loss = logits.sum() + emb.sum()
        loss.backward()

        assert spec.grad is not None
        assert pros.grad is not None
        assert hub.grad is not None

    def test_parameter_count(self):
        """Sanity check that Pipeline B is not trivially small."""
        from src.models.pipeline_b.audio_emotion_head import AudioEmotionHead

        model = AudioEmotionHead(num_classes=NUM_CLASSES, embed_dim=D)
        total = sum(p.numel() for p in model.parameters())
        assert total > 100_000, f"Pipeline B only has {total} params — too few"
