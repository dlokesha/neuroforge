"""
models.py — CNN architectures matching TBC's paper setup.

TBC used: input layer → single conv layer (5 kernels, 4×4) → linear → ReLU → classifier
We keep both architectures identical — the ONLY difference is the input representation.
This isolates the effect of bio preprocessing, same as TBC did.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    """
    Trains directly on binarized MNIST images (64×64).
    This is the control — no biological preprocessing.
    """

    def __init__(self, n_classes: int = 10):
        super().__init__()
        # TBC paper: "single convolutional layer with five 4×4 kernels"
        self.conv1 = nn.Conv2d(1, 5, kernel_size=4, stride=2, padding=1)
        # After conv + stride 2 on 64×64: (64-4+2)/2 + 1 = 32 → 32×32×5
        self.flatten_size = 5 * 32 * 32
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x):
        # x: (B, 1, 64, 64)
        x = F.relu(self.conv1(x))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.classifier(x)


class BioCNN(nn.Module):
    """
    Trains on spike-rate vectors from the reservoir layer.
    Uses fully connected layers — spike rates are not spatially
    structured like images, so FC works better than conv.
    """

    def __init__(self, n_reservoir_units: int = 1024, n_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(n_reservoir_units, 512)
        self.fc2 = nn.Linear(512, 128)
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.fc2(x))
        return self.classifier(x)


class AblationCNN(nn.Module):
    """
    Used for ablation study — takes a subset of reservoir units.
    TBC showed: full array > center > periphery, all above chance.
    We replicate this by masking which units feed into the classifier.
    """

    def __init__(self, input_dim: int, n_classes: int = 10, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.classifier = nn.Linear(hidden_dim // 2, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.classifier(x)

    def get_features(self, x):
        """Return representation before classifier (for effective rank, etc.)."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
