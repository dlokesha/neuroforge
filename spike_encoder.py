"""
spike_encoder.py — Convert images into temporal spike trains.

TBC Post 1: "Our results rely on simple spike-rate summaries which do not
capture the temporal structure known to be central to biological computation.
Ongoing work focuses on decoding spike timing and using time as an
explicit encoding dimension."

This is exactly what Part 3 builds.

Rate coding (Part 1):  image → fraction of time each unit was active
Temporal coding (Part 3): image → WHEN each neuron fires (precise timing)

Why timing matters:
  Real neurons communicate through spike timing, not just rate.
  A neuron firing at t=2ms vs t=8ms carries different information
  even if both fire once. The brain uses this — we replicate it here.

Poisson spike trains:
  The standard model for biological spike generation.
  Each pixel intensity becomes a firing RATE (spikes/second).
  We then sample actual spike times from a Poisson process.
  High intensity pixel → high firing rate → spikes come early and often
  Low intensity pixel  → low firing rate  → spikes are rare and late
"""

import numpy as np
import torch


class PoissonEncoder:
    """
    Encodes a static image into a temporal sequence of spike trains.

    Each pixel → a neuron with firing rate proportional to pixel intensity.
    We simulate T timesteps and record when each neuron fires.

    Output shape: (T, n_neurons) — binary matrix, 1 = spike, 0 = silence
    """

    def __init__(
        self,
        n_timesteps: int = 100,    # simulation duration (ms)
        max_rate: float = 100.0,   # max firing rate (Hz) for brightest pixel
        dt: float = 0.001,         # timestep size (seconds) = 1ms
        seed: int = 42,
    ):
        self.n_timesteps = n_timesteps
        self.max_rate = max_rate
        self.dt = dt
        self.rng = np.random.RandomState(seed)

    def encode(self, image: np.ndarray) -> np.ndarray:
        """
        Convert a 2D image into a spike train matrix.

        Args:
            image: (H, W) grayscale image, values in [0, 1]
        Returns:
            spikes: (n_timesteps, H*W) binary array
                    spikes[t, i] = 1 means neuron i fired at timestep t
        """
        # Flatten image to 1D — each pixel is one neuron
        pixels = image.flatten()  # (n_neurons,)
        n_neurons = len(pixels)

        # Convert pixel intensity to firing rate
        # Bright pixel (1.0) → max_rate Hz
        # Dark pixel (0.0)   → 0 Hz
        rates = pixels * self.max_rate  # (n_neurons,) in Hz

        # Probability of firing in each 1ms timestep
        # P(spike in dt) = rate * dt
        # e.g. 100Hz neuron: P = 100 * 0.001 = 0.1 (10% chance per ms)
        spike_probs = rates * self.dt  # (n_neurons,)
        spike_probs = np.clip(spike_probs, 0, 1)

        # Sample spike trains using Poisson process
        # For each timestep, each neuron independently decides to fire
        spikes = self.rng.rand(self.n_timesteps, n_neurons) < spike_probs

        return spikes.astype(np.float32)  # (n_timesteps, n_neurons)

    def encode_batch(self, images: np.ndarray) -> np.ndarray:
        """Encode a batch of images. Returns (N, T, n_neurons)."""
        return np.array([self.encode(img) for img in images])

    def get_first_spike_times(self, spikes: np.ndarray) -> np.ndarray:
        """
        Extract first spike time for each neuron — a key temporal feature.

        Neurons that fire EARLY encode high-intensity pixels.
        Neurons that never fire encode dark/silent pixels.

        Args:
            spikes: (n_timesteps, n_neurons) binary array
        Returns:
            first_spike_times: (n_neurons,) — timestep of first spike,
                               n_timesteps if neuron never fired
        """
        n_timesteps, n_neurons = spikes.shape
        first_times = np.full(n_neurons, n_timesteps, dtype=float)

        for t in range(n_timesteps):
            fired = spikes[t] > 0
            # Only update neurons that haven't fired yet
            never_fired = first_times == n_timesteps
            first_times[fired & never_fired] = t

        # Normalize to [0, 1]
        first_times = first_times / n_timesteps
        return first_times  # (n_neurons,)


class TemporalEncoder:
    """
    More sophisticated temporal encoder using rank-order coding.

    Rank-order coding: neurons fire in order of their intensity.
    Brightest pixel fires first, darkest fires last (or not at all).
    This is extremely efficient — identity of image conveyed in first few spikes.

    Discovered by Thorpe et al. (1996) — the brain uses this for rapid
    visual processing (we recognize faces in ~150ms despite slow neurons).
    """

    def __init__(self, n_timesteps: int = 100):
        self.n_timesteps = n_timesteps

    def encode(self, image: np.ndarray) -> np.ndarray:
        """
        Encode image using rank-order coding.

        Brightest pixel → fires at t=0
        Dimmest pixel   → fires at t=n_timesteps-1 (or never)
        """
        pixels = image.flatten()
        n_neurons = len(pixels)

        # Rank pixels by intensity (brightest = rank 0)
        ranks = np.argsort(np.argsort(-pixels))  # (n_neurons,)

        # Convert rank to spike time
        # Rank 0 (brightest) → t=0, Rank N-1 (dimmest) → t=T-1
        spike_times = (ranks / n_neurons * self.n_timesteps).astype(int)
        spike_times = np.clip(spike_times, 0, self.n_timesteps - 1)

        # Build spike matrix
        spikes = np.zeros((self.n_timesteps, n_neurons), dtype=np.float32)
        for neuron_idx, t in enumerate(spike_times):
            if pixels[neuron_idx] > 0.05:  # Only fire if pixel is bright enough
                spikes[t, neuron_idx] = 1.0

        return spikes  # (n_timesteps, n_neurons)

    def encode_batch(self, images: np.ndarray) -> np.ndarray:
        return np.array([self.encode(img) for img in images])
