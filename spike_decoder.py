"""
spike_decoder.py — Decode temporal spike trains into feature vectors.

Takes the rich temporal spike patterns from the encoder and extracts
meaningful features that a downstream classifier can use.

Three decoding strategies, each capturing different temporal information:

1. RateDecoder     — baseline, ignores timing (same as Part 1)
2. TemporalDecoder — uses first spike times (when neurons fire)
3. SyncDecoder     — uses synchrony patterns (which neurons fire together)

TBC's insight: timing carries MORE information than rate alone.
We prove this by comparing classifier accuracy across all three decoders.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RateDecoder:
    """
    Baseline decoder — ignores spike timing, just counts spikes.
    This is equivalent to what Part 1 did with the reservoir.
    Used as the control condition.
    """

    def decode(self, spikes: np.ndarray) -> np.ndarray:
        """
        Args:
            spikes: (n_timesteps, n_neurons) binary array
        Returns:
            features: (n_neurons,) mean firing rate per neuron
        """
        return spikes.mean(axis=0)  # (n_neurons,)

    def decode_batch(self, spike_batch: np.ndarray) -> np.ndarray:
        """Args: (N, T, n_neurons) → Returns: (N, n_neurons)"""
        return spike_batch.mean(axis=1)


class TemporalDecoder:
    """
    Temporal decoder — uses WHEN neurons fire, not just how often.

    Features extracted:
    1. First spike time — encodes intensity (bright → early)
    2. Mean inter-spike interval — encodes regularity
    3. Spike count in early window — captures onset response
    4. Spike count in late window — captures sustained response
    """

    def __init__(self, n_timesteps: int = 100, early_window: float = 0.3):
        self.n_timesteps = n_timesteps
        self.early_cutoff = int(n_timesteps * early_window)

    def decode(self, spikes: np.ndarray) -> np.ndarray:
        """
        Args:
            spikes: (n_timesteps, n_neurons)
        Returns:
            features: (4 * n_neurons,) temporal feature vector
        """
        n_timesteps, n_neurons = spikes.shape

        # Feature 1: First spike time (normalized)
        first_spike = np.full(n_neurons, 1.0)
        for t in range(n_timesteps):
            fired = spikes[t] > 0
            never_fired = first_spike == 1.0
            first_spike[fired & never_fired] = t / n_timesteps

        # Feature 2: Total spike count (normalized)
        spike_count = spikes.sum(axis=0) / n_timesteps

        # Feature 3: Early window spike density
        early_spikes = spikes[:self.early_cutoff].mean(axis=0)

        # Feature 4: Late window spike density
        late_spikes = spikes[self.early_cutoff:].mean(axis=0)

        # Concatenate all temporal features
        features = np.concatenate([
            first_spike,
            spike_count,
            early_spikes,
            late_spikes,
        ])

        return features.astype(np.float32)

    def decode_batch(self, spike_batch: np.ndarray) -> np.ndarray:
        """Args: (N, T, n_neurons) → Returns: (N, 4*n_neurons)"""
        return np.array([self.decode(s) for s in spike_batch])


class SyncDecoder:
    """
    Synchrony decoder — captures which neurons fire TOGETHER.

    Synchronous firing between neurons is a key biological signal.
    Two neurons firing at the same timestep = correlated activity = shared input.
    This captures spatial relationships that rate coding misses.

    Implementation: pairwise synchrony for neuron groups (not all pairs — too slow)
    We group neurons into bins and measure within-bin synchrony.
    """

    def __init__(self, n_timesteps: int = 100, n_groups: int = 16):
        self.n_timesteps = n_timesteps
        self.n_groups = n_groups

    def decode(self, spikes: np.ndarray) -> np.ndarray:
        """
        Args:
            spikes: (n_timesteps, n_neurons)
        Returns:
            features: (n_neurons + n_groups,)
                      rate features + synchrony features per group
        """
        n_timesteps, n_neurons = spikes.shape
        group_size = n_neurons // self.n_groups

        # Base rate features
        rate_features = spikes.mean(axis=0)

        # Synchrony features per group
        sync_features = np.zeros(self.n_groups)
        for g in range(self.n_groups):
            start = g * group_size
            end = start + group_size
            group_spikes = spikes[:, start:end]  # (T, group_size)

            # Synchrony = mean pairwise coincidence within group
            # Simplified: variance of group activity over time
            group_activity = group_spikes.sum(axis=1)  # (T,) total spikes per timestep
            sync_features[g] = group_activity.std() / (group_activity.mean() + 1e-8)

        features = np.concatenate([rate_features, sync_features])
        return features.astype(np.float32)

    def decode_batch(self, spike_batch: np.ndarray) -> np.ndarray:
        return np.array([self.decode(s) for s in spike_batch])


class SpikeClassifier(nn.Module):
    """
    Simple MLP classifier that works with any decoder's output.
    Kept identical across all decoders to isolate the effect of decoding strategy.
    """

    def __init__(self, input_dim: int, n_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.fc2(x))
        return self.classifier(x)
