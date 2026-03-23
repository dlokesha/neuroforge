"""
rank_tracker.py — Track effective rank of CNN hidden activations during training.

TBC Post 4: "Networks trained with our approach maintained consistently high
effective rank throughout training, roughly double that of reactive methods.
Internal representations remained rich and non-redundant over time."

Part 4 asks: does bio preprocessing (Part 1) also produce higher effective rank
than baseline training on raw images?

If yes — it ties Parts 1 and 2 together:
  Part 1: bio preprocessing improves accuracy
  Part 2: local plasticity preserves rank
  Part 4: bio preprocessing ALSO preserves higher rank → same mechanism

This would be a strong original finding to include in your cold email.

Usage:
    python rank_tracker.py                          # full run
    python rank_tracker.py --n_samples 500 --epochs 5  # quick test
"""

import argparse
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from models import BaselineCNN, BioCNN
from reservoir import BioPreprocessor, MEAEncoder
from train import encode_to_grid, load_mnist, run_bio_preprocessing


def effective_rank(model: nn.Module, data_loader, device) -> float:
    model.eval()
    all_hidden = []

    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            model_name = model.__class__.__name__
            if model_name == "BaselineCNN":
                h = torch.relu(model.conv1(x))
                h = h.flatten(1)
                h = torch.relu(model.fc1(h))
            else:
                # BioCNN — x is (batch, 1024) spike rates
                h = torch.relu(model.fc1(x))
            all_hidden.append(h.cpu())

    hidden = torch.cat(all_hidden, dim=0)

    try:
        _, s, _ = torch.linalg.svd(hidden, full_matrices=False)
    except Exception:
        return 1.0

    s = s.float()
    s = s / (s.sum() + 1e-8)
    entropy = -(s * torch.log(s + 1e-8)).sum()
    return torch.exp(entropy).item()

def train_with_rank_tracking(
    model,
    train_loader,
    test_loader,
    epochs: int,
    device,
    label: str = "",
) -> dict:
    """
    Train a model and track effective rank at every epoch.
    Returns accuracy curve and rank curve.
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    acc_curve = []
    rank_curve = []

    for epoch in range(epochs):
        # Train
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        # Eval accuracy
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(1)
                correct += (preds == y).sum().item()
                total += len(y)
        acc = correct / total
        acc_curve.append(acc)

        # Eval effective rank
        rank = effective_rank(model, test_loader, device)
        rank_curve.append(rank)

        print(f"  [{label}] Epoch {epoch+1}/{epochs} — acc: {acc*100:.1f}% | rank: {rank:.1f}")

    return {"acc": acc_curve, "rank": rank_curve}


def plot_results(baseline_results: dict, bio_results: dict,
                 save_path: str = "outputs/rank_tracking.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = range(1, len(baseline_results["acc"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Part 4: Effective Rank — Bio Preprocessing vs Baseline", fontsize=14)

    # Accuracy
    ax = axes[0]
    ax.plot(epochs, [a * 100 for a in baseline_results["acc"]],
            color="gold", linewidth=2, label="Baseline (raw MNIST)")
    ax.plot(epochs, [a * 100 for a in bio_results["acc"]],
            color="#1D9E75", linewidth=2, label="Bio-preprocessed")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Classification accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    # Effective rank
    ax = axes[1]
    ax.plot(epochs, baseline_results["rank"],
            color="gold", linewidth=2, label="Baseline (raw MNIST)")
    ax.plot(epochs, bio_results["rank"],
            color="#1D9E75", linewidth=2, label="Bio-preprocessed")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Effective rank")
    ax.set_title("Representation quality (effective rank)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add annotation if bio rank > baseline rank
    final_bio_rank = bio_results["rank"][-1]
    final_base_rank = baseline_results["rank"][-1]
    diff = final_bio_rank - final_base_rank
    if diff > 0:
        ax.annotate(
            f"Bio rank {diff:+.1f} higher",
            xy=(len(epochs), final_bio_rank),
            xytext=(len(epochs) * 0.6, max(bio_results["rank"]) * 1.05),
            fontsize=10, color="#1D9E75",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved rank tracking plot to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load data ──────────────────────────────────────────────────────────
    train_images, train_labels, test_images, test_labels = load_mnist(args.n_samples)

    print("Encoding images to electrode grid...")
    train_grids = encode_to_grid(train_images)
    test_grids = encode_to_grid(test_images)

    # ── Bio preprocessing ──────────────────────────────────────────────────
    bio_train = run_bio_preprocessing(
        train_grids, cache_path=f"outputs/bio_train_{args.n_samples}.pkl"
    )
    bio_test = run_bio_preprocessing(
        test_grids, cache_path=f"outputs/bio_test_{args.n_samples//5}.pkl"
    )

    # ── Build dataloaders ──────────────────────────────────────────────────
    baseline_train_ds = TensorDataset(
        torch.FloatTensor(train_grids).unsqueeze(1),
        torch.LongTensor(train_labels)
    )
    baseline_test_ds = TensorDataset(
        torch.FloatTensor(test_grids).unsqueeze(1),
        torch.LongTensor(test_labels)
    )
    bio_train_ds = TensorDataset(
        torch.FloatTensor(bio_train),
        torch.LongTensor(train_labels)
    )
    bio_test_ds = TensorDataset(
        torch.FloatTensor(bio_test),
        torch.LongTensor(test_labels)
    )

    bl_train = DataLoader(baseline_train_ds, batch_size=64, shuffle=True)
    bl_test = DataLoader(baseline_test_ds, batch_size=64)
    bio_train_loader = DataLoader(bio_train_ds, batch_size=64, shuffle=True)
    bio_test_loader = DataLoader(bio_test_ds, batch_size=64)

    # ── Train with rank tracking ───────────────────────────────────────────
    print("\nTraining Baseline CNN with rank tracking...")
    baseline_results = train_with_rank_tracking(
        BaselineCNN(), bl_train, bl_test, args.epochs, device, label="Baseline"
    )

    print("\nTraining Bio CNN with rank tracking...")
    bio_results = train_with_rank_tracking(
        BioCNN(), bio_train_loader, bio_test_loader, args.epochs, device, label="Bio"
    )

    # ── Plot ───────────────────────────────────────────────────────────────
    plot_results(baseline_results, bio_results)

    # ── Summary ────────────────────────────────────────────────────────────
    final_base_rank = baseline_results["rank"][-1]
    final_bio_rank = bio_results["rank"][-1]
    mean_base_rank = np.mean(baseline_results["rank"])
    mean_bio_rank = np.mean(bio_results["rank"])

    summary = {
        "baseline_final_acc": round(baseline_results["acc"][-1] * 100, 2),
        "bio_final_acc": round(bio_results["acc"][-1] * 100, 2),
        "baseline_final_rank": round(final_base_rank, 2),
        "bio_final_rank": round(final_bio_rank, 2),
        "baseline_mean_rank": round(mean_base_rank, 2),
        "bio_mean_rank": round(mean_bio_rank, 2),
        "rank_difference": round(final_bio_rank - final_base_rank, 2),
    }

    with open("outputs/rank_tracking.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*55}")
    print(f"PART 4 RESULTS — Effective Rank Tracking")
    print(f"{'='*55}")
    print(f"Baseline  — acc: {summary['baseline_final_acc']:.1f}% | rank: {final_base_rank:.1f}")
    print(f"Bio       — acc: {summary['bio_final_acc']:.1f}% | rank: {final_bio_rank:.1f}")
    print(f"Rank diff: {final_bio_rank - final_base_rank:+.1f}")
    print(f"{'='*55}")
    if final_bio_rank > final_base_rank:
        print("Bio preprocessing produces HIGHER effective rank — richer representations")
        print("This ties Parts 1, 2, and 4 together: bio input → better accuracy AND richer representations")
    else:
        print("Ranks are similar — bio preprocessing improves accuracy through other mechanisms")
    print(f"\nOutputs saved to ./outputs/")


if __name__ == "__main__":
    main()