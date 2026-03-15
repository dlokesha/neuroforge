"""
plasticity.py — Three learning approaches for continual learning.

Replicates TBC Post 4 comparison:

1. StandardNetwork     — plain backprop, no plasticity fix (collapses over time)
2. ContinualBackprop   — Sutton et al. reactive method (recycles dead neurons)
3. LocalPlasticityNet  — TBC's biologically inspired proactive approach

TBC's key insight: biological neurons maintain feature diversity through
LOCAL interactions — no global oversight needed. Their local plasticity
rule operates at the level of individual neurons, preserving representational
richness proactively rather than reactively fixing collapse after it happens.

TBC has not published the exact rule, but based on their description we implement
a biologically grounded approximation: local homeostatic regularization that
keeps each neuron's activation variance within a target range, preventing
both dead neurons (too silent) and saturated neurons (too active).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ── Shared network architecture ──────────────────────────────────────────────

class ContinualMLP(nn.Module):
    """
    Simple MLP used across all three methods.
    Input: 784 (flattened MNIST pixel)
    Hidden: 2 layers of 256 units
    Output: 10 classes

    TBC used this class of architecture for their Permuted MNIST experiments.
    Keeping architecture identical isolates the effect of the learning rule.
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, n_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_classes)
        self.hidden_dim = hidden_dim

    def forward(self, x, return_hidden: bool = False):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        logits = self.classifier(h2)
        if return_hidden:
            return logits, h2
        return logits

    def get_hidden(self, x):
        with torch.no_grad():
            h1 = F.relu(self.fc1(x))
            h2 = F.relu(self.fc2(h1))
        return h2


# ── Method 1: Standard backprop (baseline — will collapse) ───────────────────

class StandardNetwork(ContinualMLP):
    """
    Plain backprop. No plasticity preservation.
    Expected to show catastrophic forgetting — accuracy on old tasks
    drops rapidly as new tasks are learned.
    This is the 'control' condition in TBC's experiment.
    """

    def train_step(self, x, y, optimizer, criterion):
        optimizer.zero_grad()
        logits = self.forward(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        return loss.item()


# ── Method 2: Continual Backpropagation (Sutton et al. reactive method) ──────

class ContinualBackprop(ContinualMLP):
    """
    Reactive method from Sutton et al. (Nature 2024).
    Periodically recycles low-utility neurons by resetting their weights.

    'Utility' = how much a neuron contributes to the output over time.
    Low utility neurons get reset to random weights + bias zeroed.
    This restores plasticity AFTER collapse has begun — reactive, not proactive.

    TBC result: reaches ~96% on new tasks but shows faster forgetting
    than their local plasticity rule.
    """

    def __init__(self, *args, replacement_rate: float = 0.001, **kwargs):
        super().__init__(*args, **kwargs)
        self.replacement_rate = replacement_rate
        # Track contribution utility for each hidden unit
        self.register_buffer('utility', torch.zeros(self.hidden_dim))
        self.step_count = 0

    def train_step(self, x, y, optimizer, criterion):
        optimizer.zero_grad()
        logits, h2 = self.forward(x, return_hidden=True)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        # Update utility: contribution = mean absolute activation × mean absolute outgoing weight
        with torch.no_grad():
            activation_contrib = h2.abs().mean(dim=0)
            weight_contrib = self.classifier.weight.abs().mean(dim=0)
            self.utility = 0.99 * self.utility + 0.01 * (activation_contrib * weight_contrib)

        self.step_count += 1

        # Recycle lowest-utility neurons periodically
        if self.step_count % 100 == 0:
            self._recycle_neurons()

        return loss.item()

    def _recycle_neurons(self):
        """Reset weights of lowest-utility neurons in fc2."""
        n_replace = max(1, int(self.replacement_rate * self.hidden_dim))
        lowest = self.utility.argsort()[:n_replace]

        with torch.no_grad():
            # Reset incoming weights to fc2 for these neurons
            nn.init.kaiming_normal_(self.fc2.weight[lowest])
            self.fc2.bias[lowest].zero_()
            # Reset their utility scores
            self.utility[lowest] = self.utility.mean()


# ── Method 3: Local Plasticity Rule (TBC's approach) ─────────────────────────

class LocalPlasticityNet(ContinualMLP):
    """
    TBC's biologically inspired proactive plasticity rule.

    Key difference from Continual Backprop:
    - PROACTIVE: acts continuously during training, not after collapse
    - LOCAL: each neuron adjusts based only on its own activity — no global utility
    - BIOLOGICAL: mirrors homeostatic plasticity in real neurons

    TBC description: "local interactions between neurons that naturally
    maintain feature diversity over time... operates without global oversight.
    Each neuron adjusts its connections based solely on local activity patterns."

    Our implementation: homeostatic regularization that targets a specific
    activation variance per neuron. Neurons that are too silent get boosted,
    neurons that are too saturated get dampened. This directly mirrors the
    biological mechanism TBC references.

    TBC result: ~97% on new tasks, ~2× effective rank vs reactive methods,
    significantly better retention of past tasks.
    """

    def __init__(
        self,
        *args,
        target_variance: float = 0.1,   # target activation variance per neuron
        plasticity_lr: float = 0.01,    # how fast homeostatic correction applies
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.target_variance = target_variance
        self.plasticity_lr = plasticity_lr

        # Running estimate of each neuron's activation variance
        self.register_buffer('running_var', torch.ones(self.hidden_dim) * target_variance)
        self.register_buffer('running_mean', torch.zeros(self.hidden_dim))

    def train_step(self, x, y, optimizer, criterion):
        optimizer.zero_grad()
        logits, h2 = self.forward(x, return_hidden=True)
        loss = criterion(logits, y)

        # Local plasticity regularization loss
        # Penalizes deviation from target activation variance — locally per neuron
        batch_var = h2.var(dim=0)
        batch_mean = h2.mean(dim=0)

        # Update running stats (exponential moving average)
        with torch.no_grad():
            self.running_var = 0.95 * self.running_var + 0.05 * batch_var.detach()
            self.running_mean = 0.95 * self.running_mean + 0.05 * batch_mean.detach()

        # Homeostatic loss: push each neuron's variance toward target
        # This is the LOCAL plasticity rule — each neuron independently
        homeostatic_loss = ((batch_var - self.target_variance) ** 2).mean()

        # Diversity loss: discourage neurons from becoming redundant
        # Correlations between neurons should be low — mirrors biological feature diversity
        if h2.shape[0] > 1:
            h2_norm = F.normalize(h2, dim=0)
            correlation = (h2_norm.T @ h2_norm) / h2.shape[0]
            # Penalize off-diagonal correlations
            eye = torch.eye(self.hidden_dim, device=h2.device)
            diversity_loss = ((correlation - eye) ** 2).mean()
        else:
            diversity_loss = torch.tensor(0.0, device=h2.device)

        total_loss = loss + self.plasticity_lr * homeostatic_loss + self.plasticity_lr * diversity_loss
        total_loss.backward()
        optimizer.step()

        return loss.item()


# ── Effective rank metric ─────────────────────────────────────────────────────

def effective_rank(hidden_activations: torch.Tensor) -> float:
    """
    Compute effective rank of hidden layer activations.

    TBC Figure 2: Local plasticity rule maintains ~2× higher effective rank
    than reactive methods throughout training.

    Effective rank = exp(entropy of normalized singular value distribution)
    High rank → rich, diverse representations
    Low rank → collapsed, redundant representations (plasticity loss)

    Reference: Roy & Vetterli (2007) "The effective rank: A measure of
    effective dimensionality"
    """
    with torch.no_grad():
        # SVD of activation matrix
        try:
            _, s, _ = torch.linalg.svd(hidden_activations, full_matrices=False)
        except Exception:
            return 1.0

        # Normalize to get probability distribution
        s = s.float()
        s = s / (s.sum() + 1e-8)

        # Entropy of singular value distribution
        entropy = -(s * torch.log(s + 1e-8)).sum()
        return torch.exp(entropy).item()


# ── EWC (for continual_train.py compatibility) ─────────────────────────────

def get_flat_params(model: nn.Module) -> torch.Tensor:
    """Return all model parameters as a single flat vector."""
    return torch.cat([p.data.flatten() for p in model.parameters()])


def set_flat_params(model: nn.Module, flat: torch.Tensor):
    """Set model parameters from a flat vector (in-place)."""
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[offset : offset + n].view_as(p))
        offset += n


def compute_fisher_diagonal(model: nn.Module, dataloader, device, n_samples: int = 500):
    """Approximate diagonal Fisher for EWC. Returns flat tensor same size as get_flat_params(model)."""
    model.eval()
    fisher = None
    n_seen = 0
    for x, y in dataloader:
        if n_seen >= n_samples:
            break
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        logits = model(x)
        log_probs = torch.log_softmax(logits, dim=1)
        nll = -log_probs[range(len(y)), y].sum()
        nll.backward()
        if fisher is None:
            fisher = torch.cat([p.grad.data.flatten().pow(2) for p in model.parameters()])
        else:
            fisher += torch.cat([p.grad.data.flatten().pow(2) for p in model.parameters()])
        n_seen += x.size(0)
    if n_seen > 0:
        fisher = fisher / n_seen
    return fisher


class EWCRegularizer:
    """
    Elastic Weight Consolidation: penalize moving away from old task parameters.
    Used by continual_train.py for the 'Continual backprop' (EWC) method.
    """

    def __init__(self, model: nn.Module, dataloader, device, lambda_ewc: float = 1000.0, n_fisher_samples: int = 500):
        self.lambda_ewc = lambda_ewc
        self.old_params = get_flat_params(model).detach().clone()
        self.fisher = compute_fisher_diagonal(model, dataloader, device, n_fisher_samples)
        if self.fisher is not None:
            self.fisher = self.fisher.to(device)

    def penalty(self, model: nn.Module) -> torch.Tensor:
        if self.fisher is None:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        curr = get_flat_params(model)
        return (self.fisher * (curr - self.old_params).pow(2)).sum() * (self.lambda_ewc / 2.0)
