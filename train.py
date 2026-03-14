"""
train.py — Run the full TBC experiment comparison.

Usage:
    python train.py                     # full run
    python train.py --n_samples 500     # quick smoke test
    python train.py --epochs 5          # fast iteration

What this does:
    1. Loads MNIST, binarizes it, resizes to 64x64
    2. Runs all images through the BioPreprocessor (reservoir layer)
    3. Trains BaselineCNN on raw images
    4. Trains BioCNN on bio-processed spike rates
    5. Plots accuracy curves — should show bio beats baseline
    6. Saves results to outputs/
"""

import argparse
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from models import AblationCNN, BaselineCNN, BioCNN
from reservoir import BioPreprocessor, MEAEncoder


def load_mnist(n_samples: int = 2000, data_dir: str = "./data"):
    """Load MNIST and binarize — matching TBC's preprocessing exactly."""
    print("Loading MNIST...")
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    # Subsample
    train_images = train_data.data[:n_samples].numpy() / 255.0
    train_labels = train_data.targets[:n_samples].numpy()
    test_images = test_data.data[: n_samples // 5].numpy() / 255.0
    test_labels = test_data.targets[: n_samples // 5].numpy()

    # Binarize: white digits on black background
    train_images = (train_images > 0.5).astype(float)
    test_images = (test_images > 0.5).astype(float)

    return train_images, train_labels, test_images, test_labels


def encode_to_grid(images: np.ndarray, grid_size: int = 64) -> np.ndarray:
    """Resize binarized MNIST to 64×64 electrode grid."""
    encoder = MEAEncoder(grid_size=grid_size)
    encoded = np.array([encoder.encode(img) for img in images])
    return encoded  # (N, 64, 64)


def run_bio_preprocessing(images: np.ndarray, cache_path: str = None) -> np.ndarray:
    """
    Run all images through the reservoir.
    Caches results to disk — reservoir is slow, don't rerun unnecessarily.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached bio representations from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("Running bio preprocessing (this takes a few minutes)...")
    t0 = time.time()
    preprocessor = BioPreprocessor(n_reservoir_units=1024, grid_size=64, steps=10)
    bio_reps = preprocessor.process_batch(images)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s — {len(images)/elapsed:.1f} images/sec")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(bio_reps, f)
        print(f"Cached to {cache_path}")

    return bio_reps  # (N, 1024)


def train_model(model, train_loader, test_loader, epochs: int, lr: float = 1e-3, label: str = ""):
    """Train a model and return per-epoch accuracy on both splits."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_accs, test_accs = [], []

    for epoch in range(epochs):
        # Train
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        # Eval train
        model.eval()
        with torch.no_grad():
            correct = total = 0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                preds = model(x_batch).argmax(1)
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)
            train_accs.append(correct / total)

            # Eval test
            correct = total = 0
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                preds = model(x_batch).argmax(1)
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)
            test_accs.append(correct / total)

        print(f"  [{label}] Epoch {epoch+1}/{epochs} — test acc: {test_accs[-1]*100:.1f}%")

    return train_accs, test_accs


def run_ablation(
    bio_train: np.ndarray,
    bio_test: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    n_units: int = 1024,
    epochs: int = 10,
):
    """
    Ablation study replicating TBC Figure 4 (right panel).

    Splits reservoir units into:
    - Whole:      all 1024 units
    - Center:     first 512 units (approximates directly stimulated region)
    - Periphery:  last 512 units (approximates spread-beyond-stimulation)

    TBC result: Whole ≈ Center >> Periphery > chance (10%)
    All three should beat chance — proving bio spread carries real info.
    """
    print("\nRunning ablation study...")
    results = {}

    splits = {
        "Whole": (0, n_units),
        "Center": (0, n_units // 2),
        "Periphery": (n_units // 2, n_units),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for name, (start, end) in splits.items():
        subset_train = bio_train[:, start:end]
        subset_test = bio_test[:, start:end]
        input_dim = end - start

        train_ds = TensorDataset(
            torch.FloatTensor(subset_train), torch.LongTensor(train_labels)
        )
        test_ds = TensorDataset(
            torch.FloatTensor(subset_test), torch.LongTensor(test_labels)
        )
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=64)

        model = AblationCNN(input_dim=input_dim)
        _, test_accs = train_model(model, train_loader, test_loader, epochs, label=name)
        results[name] = test_accs[-1]
        print(f"  {name}: {test_accs[-1]*100:.1f}%")

    return results


def plot_results(baseline_test, bio_test, save_path: str = "outputs/accuracy_curves.png"):
    """
    Reproduce TBC Figure 4 (left panel):
    Bio accuracy curve should be higher and converge faster than baseline.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(baseline_test) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("TBC Replication: Biological Preprocessing vs Baseline", fontsize=14)

    # Accuracy curves
    ax = axes[0]
    ax.plot(epochs, [a * 100 for a in baseline_test], "gold", linewidth=2, label="Baseline (raw MNIST)")
    ax.plot(epochs, [a * 100 for a in bio_test], "#1D9E75", linewidth=2, label="Bio-preprocessed")
    ax.axhline(y=10, color="gray", linestyle="--", alpha=0.5, label="Chance (10%)")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Classification accuracy over training")
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # Final accuracy bar
    ax = axes[1]
    final_baseline = baseline_test[-1] * 100
    final_bio = bio_test[-1] * 100
    bars = ax.bar(
        ["Baseline", "Bio-preprocessed"],
        [final_baseline, final_bio],
        color=["gold", "#1D9E75"],
        width=0.5,
    )
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_ylabel("Final test accuracy (%)")
    ax.set_title(f"Final accuracy (Δ = {final_bio - final_baseline:+.1f}%)")
    ax.set_ylim(0, 100)
    ax.axhline(y=10, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved accuracy plot to {save_path}")
    plt.show()


def plot_activation_spread(
    raw_image: np.ndarray,
    electrode_grid: np.ndarray,
    spike_readout: np.ndarray,
    save_path: str = "outputs/activation_spread.png",
):
    """
    Visualize TBC Figure 3:
    Shows how activity spreads beyond the stimulated region.
    Raw digit → electrode grid → reservoir spread.
    """
    from reservoir import BioPreprocessor

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Get spatial readout
    preprocessor = BioPreprocessor(n_reservoir_units=1024)
    spatial = preprocessor.reservoir.get_spatial_readout(spike_readout, grid_size=64)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Neural activity spread beyond stimulation region", fontsize=13)

    axes[0].imshow(raw_image, cmap="gray")
    axes[0].set_title("Original MNIST digit")
    axes[0].axis("off")

    axes[1].imshow(electrode_grid, cmap="gray")
    axes[1].set_title("Electrode grid (stimulation pattern)")
    axes[1].axis("off")
    # Red box showing original 28×28 digit area
    rect = plt.Rectangle(
        (18, 18), 28, 28, linewidth=2, edgecolor="red", facecolor="none"
    )
    axes[1].add_patch(rect)
    axes[1].text(22, 15, "28×28 digit area", color="red", fontsize=8)

    axes[2].imshow(spatial, cmap="hot")
    axes[2].set_title("Reservoir activity spread")
    axes[2].axis("off")
    rect2 = plt.Rectangle(
        (18, 18), 28, 28, linewidth=2, edgecolor="cyan", facecolor="none"
    )
    axes[2].add_patch(rect2)
    axes[2].text(22, 15, "original region", color="cyan", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved activation spread plot to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=2000, help="Training samples (500 for smoke test)")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--skip_ablation", action="store_true", help="Skip ablation study")
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)

    # ── 1. Load and encode data ─────────────────────────────────────────────
    train_images, train_labels, test_images, test_labels = load_mnist(args.n_samples)

    print("Encoding images to electrode grid...")
    train_grids = encode_to_grid(train_images)   # (N, 64, 64)
    test_grids = encode_to_grid(test_images)

    # ── 2. Bio preprocessing ────────────────────────────────────────────────
    bio_train = run_bio_preprocessing(
        train_grids, cache_path=f"outputs/bio_train_{args.n_samples}.pkl"
    )
    bio_test = run_bio_preprocessing(
        test_grids, cache_path=f"outputs/bio_test_{args.n_samples//5}.pkl"
    )

    # ── 3. Build dataloaders ─────────────────────────────────────────────────
    # Baseline: raw 64×64 grids as images
    baseline_train_ds = TensorDataset(
        torch.FloatTensor(train_grids).unsqueeze(1),  # (N, 1, 64, 64)
        torch.LongTensor(train_labels),
    )
    baseline_test_ds = TensorDataset(
        torch.FloatTensor(test_grids).unsqueeze(1),
        torch.LongTensor(test_labels),
    )

    # Bio: spike rate vectors
    bio_train_ds = TensorDataset(
        torch.FloatTensor(bio_train),  # (N, 1024)
        torch.LongTensor(train_labels),
    )
    bio_test_ds = TensorDataset(
        torch.FloatTensor(bio_test),
        torch.LongTensor(test_labels),
    )

    batch_size = 64
    baseline_train_loader = DataLoader(baseline_train_ds, batch_size=batch_size, shuffle=True)
    baseline_test_loader = DataLoader(baseline_test_ds, batch_size=batch_size)
    bio_train_loader = DataLoader(bio_train_ds, batch_size=batch_size, shuffle=True)
    bio_test_loader = DataLoader(bio_test_ds, batch_size=batch_size)

    # ── 4. Train both models ─────────────────────────────────────────────────
    print("\nTraining baseline CNN on raw MNIST...")
    baseline_model = BaselineCNN()
    _, baseline_test = train_model(
        baseline_model, baseline_train_loader, baseline_test_loader,
        args.epochs, label="Baseline"
    )

    print("\nTraining Bio CNN on reservoir spike rates...")
    bio_model = BioCNN()
    _, bio_test = train_model(
        bio_model, bio_train_loader, bio_test_loader,
        args.epochs, label="Bio"
    )

    # ── 5. Plot accuracy curves ──────────────────────────────────────────────
    plot_results(baseline_test, bio_test)

    # ── 6. Activation spread visualization ──────────────────────────────────
    print("\nGenerating activation spread visualization...")
    sample_idx = 0
    plot_activation_spread(
        raw_image=train_images[sample_idx],
        electrode_grid=train_grids[sample_idx],
        spike_readout=bio_train[sample_idx],
    )

    # ── 7. Ablation study ────────────────────────────────────────────────────
    if not args.skip_ablation:
        ablation_results = run_ablation(
            bio_train, bio_test, train_labels, test_labels, epochs=args.epochs
        )
        print("\nAblation results:")
        for k, v in ablation_results.items():
            print(f"  {k}: {v*100:.1f}%")
        print("  Chance: 10.0%")

    # ── 8. Save summary ──────────────────────────────────────────────────────
    final_baseline = baseline_test[-1] * 100
    final_bio = bio_test[-1] * 100
    summary = {
        "n_samples": args.n_samples,
        "epochs": args.epochs,
        "baseline_final_acc": final_baseline,
        "bio_final_acc": final_bio,
        "improvement": final_bio - final_baseline,
        "baseline_curve": baseline_test,
        "bio_curve": bio_test,
    }
    import json
    with open("outputs/results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"Baseline final accuracy:     {final_baseline:.1f}%")
    print(f"Bio-preprocessed accuracy:   {final_bio:.1f}%")
    print(f"Improvement:                 {final_bio - final_baseline:+.1f}%")
    print(f"(TBC paper reported: +4.7%)")
    print(f"{'='*50}")
    print(f"\nOutputs saved to ./outputs/")


if __name__ == "__main__":
    main()
