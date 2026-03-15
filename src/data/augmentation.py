"""Audio and spectrogram augmentation for training robustness.

Applied **on-the-fly** during training (not cached).

Augmentations:
1. **AudioAugmentor** — waveform-level: additive noise, time stretch, pitch shift.
2. **SpecAugment** — spectrogram-level: frequency and time masking.

**Important:** Prosodic contours are **NOT** augmented—they must reflect
the true prosody of the utterance so the TCN learns genuine patterns.
"""

from __future__ import annotations

import numpy as np


class AudioAugmentor:
    """Probabilistic waveform-level augmentations.

    Each augmentation is applied independently with its own probability.
    """

    def __init__(
        self,
        noise_prob: float = 0.5,
        noise_snr_range: tuple = (10, 30),
        time_stretch_prob: float = 0.3,
        time_stretch_range: tuple = (0.9, 1.1),
        pitch_shift_prob: float = 0.3,
        pitch_shift_range: tuple = (-2, 2),
        sr: int = 16000,
    ):
        self.noise_prob = noise_prob
        self.noise_snr_range = noise_snr_range
        self.time_stretch_prob = time_stretch_prob
        self.time_stretch_range = time_stretch_range
        self.pitch_shift_prob = pitch_shift_prob
        self.pitch_shift_range = pitch_shift_range
        self.sr = sr

    def __call__(self, audio: np.ndarray, sr: int | None = None) -> np.ndarray:
        """Apply random augmentations to a waveform.

        Args:
            audio: 1-D float32 audio waveform.
            sr: Sample rate override (defaults to ``self.sr``).

        Returns:
            Augmented float32 waveform, re-normalised to [-1, 1].
        """
        import librosa  # Lazy import for speed when not augmenting

        sr = sr or self.sr
        aug = audio.copy()

        # 1. Additive Gaussian noise
        if np.random.random() < self.noise_prob:
            snr_db = np.random.uniform(*self.noise_snr_range)
            audio_power = np.mean(aug ** 2) + 1e-10
            noise_power = audio_power / (10 ** (snr_db / 10))
            noise = np.random.randn(*aug.shape).astype(np.float32)
            aug = aug + noise * np.sqrt(noise_power)

        # 2. Time stretch
        if np.random.random() < self.time_stretch_prob:
            rate = np.random.uniform(*self.time_stretch_range)
            aug = librosa.effects.time_stretch(aug, rate=rate)

        # 3. Pitch shift
        if np.random.random() < self.pitch_shift_prob:
            n_steps = np.random.uniform(*self.pitch_shift_range)
            aug = librosa.effects.pitch_shift(aug, sr=sr, n_steps=n_steps)

        # Re-normalise to [-1, 1]
        peak = np.max(np.abs(aug))
        if peak > 0:
            aug = aug / peak

        return aug.astype(np.float32)


class SpecAugment:
    """SpecAugment: frequency and time masking on log-Mel spectrograms.

    Reference: Park et al., "SpecAugment: A Simple Data Augmentation
    Method for Automatic Speech Recognition", Interspeech 2019.
    """

    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 50,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
        mask_value: float = 0.0,
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.mask_value = mask_value

    def __call__(self, spec: np.ndarray) -> np.ndarray:
        """Apply SpecAugment masking.

        Args:
            spec: Spectrogram of shape ``(n_mels, n_frames)``.

        Returns:
            Augmented spectrogram (copy — does not modify input).
        """
        spec = spec.copy()
        n_mels, n_frames = spec.shape

        # Frequency masking
        for _ in range(self.n_freq_masks):
            f = np.random.randint(0, min(self.freq_mask_param, n_mels))
            f0 = np.random.randint(0, max(1, n_mels - f))
            spec[f0 : f0 + f, :] = self.mask_value

        # Time masking
        for _ in range(self.n_time_masks):
            t = np.random.randint(0, min(self.time_mask_param, n_frames))
            t0 = np.random.randint(0, max(1, n_frames - t))
            spec[:, t0 : t0 + t] = self.mask_value

        return spec


def build_augmentor_from_config(config) -> AudioAugmentor:
    """Build an AudioAugmentor from an AugmentationConfig dataclass.

    Args:
        config: An ``AugmentationConfig`` instance (from ``src.utils.config``).

    Returns:
        Configured AudioAugmentor.
    """
    return AudioAugmentor(
        noise_prob=config.noise_prob,
        noise_snr_range=tuple(config.noise_snr_range),
        time_stretch_prob=config.time_stretch_prob,
        time_stretch_range=tuple(config.time_stretch_range),
        pitch_shift_prob=config.pitch_shift_prob,
        pitch_shift_range=tuple(config.pitch_shift_range),
    )


def build_spec_augment_from_config(config) -> SpecAugment:
    """Build a SpecAugment from an AugmentationConfig dataclass.

    Args:
        config: An ``AugmentationConfig`` instance.

    Returns:
        Configured SpecAugment.
    """
    sa = config.spec_augment
    return SpecAugment(
        freq_mask_param=sa.freq_mask_param,
        time_mask_param=sa.time_mask_param,
        n_freq_masks=sa.n_freq_masks,
        n_time_masks=sa.n_time_masks,
    )
