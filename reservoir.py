"""
reservoir.py — Software simulation of TBC's MEA pipeline.

Real TBC:  image → electrical stimulation → dish of neurons → spike readout → CNN
This file:  image → electrode encoding → Echo State Network → spike readout → CNN

The ESN (Echo State Network) is our stand-in for the biological neural culture.
It's a recurrent network with FIXED random weights — we never train it.
Just like real neurons, it spreads input through recurrent dynamics and
produces richer, higher-dimensional representations than the raw input.
"""

import numpy as np
import torch


class MEAEncoder:
    """
    Encodes a 2D image onto a simulated 64x64 multielectrode array.
    White pixels → active electrodes (stimulation sites).
    Black pixels → silent.

    TBC paper: "MNIST digits were spatially centered on a 64×64 MEA
    and stimulation was delivered to electrodes corresponding to white pixels."
    """

    def __init__(self, grid_size: int = 64):
        self.grid_size = grid_size

    def encode(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: (H, W) grayscale image, values in [0, 1]
        Returns:
            electrode_pattern: (grid_size, grid_size) binary array
        """
        from PIL import Image as PILImage

        # Resize to grid
        pil_img = PILImage.fromarray((image * 255).astype(np.uint8))
        pil_img = pil_img.resize((self.grid_size, self.grid_size), PILImage.NEAREST)
        electrode_pattern = np.array(pil_img) / 255.0

        # Binarize — white pixels become stimulation sites
        electrode_pattern = (electrode_pattern > 0.5).astype(float)
        return electrode_pattern


class ReservoirLayer:
    """
    Echo State Network — our software neuron dish.

    Key properties that mirror real neurons:
    - Fixed random recurrent weights (neurons wire themselves, we don't train this)
    - Spectral radius < 1 (stability, like real neural cultures)
    - Input scaling controls how strongly stimulation drives the network
    - Output spreads BEYOND the directly stimulated region (recurrence does this)

    TBC paper: "neural activity extended beyond electrodes directly stimulated,
    indicating that recurrent biological dynamics distributed information."
    """

    def __init__(
        self,
        n_units: int = 1024,
        spectral_radius: float = 0.9,
        input_scaling: float = 0.1,
        sparsity: float = 0.1,
        seed: int = 42,
    ):
        self.n_units = n_units
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.seed = seed

        rng = np.random.RandomState(seed)

        # Recurrent weight matrix — sparse random, like synaptic connections
        W = rng.randn(n_units, n_units)
        mask = rng.rand(n_units, n_units) > sparsity
        W[mask] = 0

        # Scale spectral radius so the network doesn't explode
        eigenvalues = np.linalg.eigvals(W)
        W = W * (spectral_radius / np.max(np.abs(eigenvalues)))
        self.W = W

        # Input weights — maps electrode grid to reservoir units
        n_electrodes = 64 * 64
        self.W_in = rng.randn(n_units, n_electrodes) * input_scaling

        self.state = np.zeros(n_units)

    def reset(self):
        self.state = np.zeros(self.n_units)

    def step(self, electrode_pattern: np.ndarray) -> np.ndarray:
        """
        Single timestep: feed stimulation pattern, get reservoir state back.
        Uses tanh activation — standard for ESNs, approximates neural firing.
        """
        u = electrode_pattern.flatten()
        self.state = np.tanh(self.W_in @ u + self.W @ self.state)
        return self.state.copy()

    def stimulate(self, electrode_pattern: np.ndarray, steps: int = 10) -> np.ndarray:
        """
        Run multiple timesteps of stimulation and return spike-rate summary.

        TBC paper: "we recorded activity over a short time window and summarized
        it using spike-rate measurements."

        Spike rate ≈ mean activation magnitude over the stimulation window.
        We track which reservoir units were above threshold — "spiking."
        """
        self.reset()
        activations = []

        for _ in range(steps):
            state = self.step(electrode_pattern)
            activations.append(state)

        activations = np.array(activations)  # (steps, n_units)

        # Spike rate readout: fraction of time each unit was "active"
        threshold = 0.1
        spike_rates = (np.abs(activations) > threshold).mean(axis=0)

        return spike_rates  # (n_units,)

    def get_spatial_readout(self, spike_rates: np.ndarray, grid_size: int = 64) -> np.ndarray:
        """
        Project reservoir activity back onto a 2D spatial grid.
        This lets us visualize where activity "spread" — the key TBC finding.
        Uses random projection (same seed = reproducible mapping).
        """
        rng = np.random.RandomState(self.seed + 1)
        projection = rng.randn(grid_size * grid_size, self.n_units) * 0.1
        spatial = projection @ spike_rates
        spatial = spatial.reshape(grid_size, grid_size)
        # Normalize to [0,1] for visualization
        spatial = (spatial - spatial.min()) / (spatial.max() - spatial.min() + 1e-8)
        return spatial


class BioPreprocessor:
    """
    Full pipeline: image → MEA encoding → reservoir → spike readout.
    Drop-in wrapper used by the training script.
    """

    def __init__(self, n_reservoir_units: int = 1024, grid_size: int = 64, steps: int = 10):
        self.encoder = MEAEncoder(grid_size=grid_size)
        self.reservoir = ReservoirLayer(n_units=n_reservoir_units)
        self.grid_size = grid_size
        self.steps = steps

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: (H, W) grayscale numpy array, values in [0, 1]
        Returns:
            spike_rates: (n_reservoir_units,) float array
        """
        electrode_pattern = self.encoder.encode(image)
        spike_rates = self.reservoir.stimulate(electrode_pattern, steps=self.steps)
        return spike_rates

    def process_batch(self, images: np.ndarray) -> np.ndarray:
        """Process a batch of images. Returns (N, n_units) array."""
        results = []
        for i, img in enumerate(images):
            if i % 100 == 0:
                print(f"  Bio-processing {i}/{len(images)}...", flush=True)
            results.append(self.process(img))
        return np.array(results)
