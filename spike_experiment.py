"""
spike_experiment.py — Compare rate coding vs temporal coding on MNIST.

TBC's stated next frontier: "Ongoing work focuses on decoding spike timing
and using time as an explicit encoding dimension."

This experiment directly addresses that — we build what they're working on.

Three conditions:
  1. Rate coding     — baseline (what Part 1 did)
  2. Temporal coding — first spike times + onset/offset responses
  3. Sync coding     — rate + synchrony between neuron groups

Expected result: temporal > rate > sync (at low timesteps)
"""

import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from spike_decoder import RateDecoder, SpikeClassifier, SyncDecoder, TemporalDecoder
from spike_encoder import PoissonEncoder


def load_mnist(n_samples: int = 1000, data_dir: str = "./data"):
    print("Loading MNIST...")
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_images = train_data.data[:n_samples].numpy() / 255.0
    train_labels = train_data.targets[:n_samples].numpy()
    test_images = test_data.data[:n_samples // 5].numpy() / 255.0
    test_labels = test_data.targets[:n_samples // 5].numpy()

    return train_images, train_labels, test_images, test_labels


def encode_dataset(images, encoder, decoder, label: str):
    print(f"  Encoding {len(images)} images with {label}...")
    t0 = time.time()
    spike_trains = encoder.encode_batch(images)
    features = decoder.decode_batch(spike_trains)
    print(f"  Done in {time.time()-t0:.1f}s — feature shape: {features.shape}")
    return features


def train_classifier(features_train, labels_train, features_test, labels_test,
                     epochs: int = 10, label: str = "") -> list:
    input_dim = features_train.shape[1]
    model = SpikeClassifier(input_dim=input_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_ds = TensorDataset(
        torch.FloatTensor(features_train),
        torch.LongTensor(labels_train)
    )
    test_ds = TensorDataset(
        torch.FloatTensor(features_test),
        torch.LongTensor(labels_test)
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)

    test_accs = []
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(1)
                correct += (preds == y).sum().item()
                total += len(y)
        acc = correct / total
        test_accs.append(acc)
        print(f"  [{label}] Epoch {epoch+1}/{epochs} — {acc*100:.1f}%")

    return test_accs


def plot_results(results: dict, save_path: str = "outputs/spike_results.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    colors = {
        "Rate coding (baseline)": "gold",
        "Temporal coding": "#1D9E75",
        "Sync coding": "#7F77DD",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Part 3: Rate vs Temporal Spike Coding on MNIST", fontsize=14)

    ax = axes[0]
    for name, accs in results.items():
        ax.plot([a * 100 for a in accs], color=colors.get(name, "gray"),
                linewidth=2, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy over training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    ax = axes[1]
    names = list(results.keys())
    final_accs = [results[n][-1] * 100 for n in names]
    bar_colors = [colors.get(n, "gray") for n in names]
    bars = ax.bar(range(len(names)), final_accs, color=bar_colors, width=0.5)
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.split(" ")[0] for n in names])
    ax.set_ylabel("Final accuracy (%)")
    ax.set_title("Final accuracy comparison")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n_timesteps", type=int, default=100)
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)

    # Load data
    train_images, train_labels, test_images, test_labels = load_mnist(args.n_samples)

    # Build encoder
    encoder = PoissonEncoder(n_timesteps=args.n_timesteps)

    # Three decoders
    experiments = [
        ("Rate coding (baseline)", RateDecoder()),
        ("Temporal coding", TemporalDecoder(n_timesteps=args.n_timesteps)),
        ("Sync coding", SyncDecoder(n_timesteps=args.n_timesteps)),
    ]

    results = {}
    summary = []

    for name, decoder in experiments:
        print(f"\n{'='*50}")
        print(f"Running: {name}")
        print(f"{'='*50}")

        # Encode
        train_features = encode_dataset(train_images, encoder, decoder, name)
        test_features = encode_dataset(test_images, encoder, decoder, name)

        # Train
        accs = train_classifier(
            train_features, train_labels,
            test_features, test_labels,
            epochs=args.epochs, label=name
        )
        results[name] = accs
        summary.append({
            "method": name,
            "final_acc": round(accs[-1] * 100, 2),
            "feature_dim": train_features.shape[1],
        })

    # Plot
    plot_results(results)

    # Save summary
    with open("outputs/spike_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print table
    print(f"\n{'='*55}")
    print(f"{'Method':<25} {'Feature dim':>12} {'Final acc':>10}")
    print(f"{'='*55}")
    for s in summary:
        print(f"{s['method']:<25} {s['feature_dim']:>12} {s['final_acc']:>9.1f}%")
    print(f"{'='*55}")
    print(f"\nTBC next step: temporal coding captures what rate coding misses")
    print(f"Outputs saved to ./outputs/")


if __name__ == "__main__":
    main()
