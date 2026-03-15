"""
continual_train.py — Run the full TBC Post 4 continual learning experiment.

Usage:
    python continual_train.py                    # full 200 tasks
    python continual_train.py --n_tasks 20       # quick smoke test
    python continual_train.py --n_tasks 50       # medium run (~5 min)

Replicates TBC Figures 1, 2, 3:
    Figure 1 — Plasticity: accuracy on newly introduced tasks over time
    Figure 2 — Effective rank: representation quality over time
    Figure 3 — Retention: accuracy on ALL past tasks over time
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from permuted_mnist import PermutedMNIST
from plasticity import (
    ContinualBackprop,
    LocalPlasticityNet,
    StandardNetwork,
    effective_rank,
)


def evaluate(model, test_loader, device) -> float:
    """Evaluate model accuracy on a single task."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += len(y)
    return correct / total


def evaluate_retention(model, test_loaders: list, device) -> float:
    """Average accuracy across all past tasks — measures forgetting."""
    accs = [evaluate(model, loader, device) for loader in test_loaders]
    return np.mean(accs)


def compute_effective_rank(model, benchmark, task_id: int, device) -> float:
    """Compute effective rank of hidden activations on current task."""
    _, test_loader = benchmark.get_task(task_id)
    model.eval()
    all_hidden = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            h = model.get_hidden(x)
            all_hidden.append(h)
    hidden = torch.cat(all_hidden, dim=0)
    return effective_rank(hidden)


def train_one_task(model, train_loader, device, epochs: int = 1) -> list:
    """Train model on one task for a given number of epochs."""
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model.train()
    losses = []

    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = model.train_step(x, y, optimizer, criterion)
            losses.append(loss)

    return losses


def run_experiment(
    model,
    benchmark: PermutedMNIST,
    n_tasks: int,
    label: str,
    device,
    eval_retention_every: int = 10,
    epochs_per_task: int = 3,
) -> dict:
    """
    Run full continual learning experiment across n_tasks.
    Returns dict with plasticity, rank, and retention curves.
    """
    plasticity_curve = []    # acc on current (new) task
    rank_curve = []          # effective rank after each task
    retention_curve = []     # avg acc on all past tasks
    retention_tasks = []     # which tasks retention was measured at

    print(f"\n{'='*50}")
    print(f"Running: {label}")
    print(f"{'='*50}")

    for task_id in range(n_tasks):
        train_loader, test_loader = benchmark.get_task(task_id)

        # Train on current task
        train_one_task(model, train_loader, device, epochs=epochs_per_task)

        # Plasticity: accuracy on the task just learned
        acc = evaluate(model, test_loader, device)
        plasticity_curve.append(acc)

        # Effective rank
        rank = compute_effective_rank(model, benchmark, task_id, device)
        rank_curve.append(rank)

        # Retention: every N tasks, evaluate on all past tasks
        if task_id % eval_retention_every == 0 or task_id == n_tasks - 1:
            all_test_loaders = benchmark.get_all_test_loaders(task_id)
            retention = evaluate_retention(model, all_test_loaders, device)
            retention_curve.append(retention)
            retention_tasks.append(task_id)
            print(
                f"  Task {task_id+1:3d}/{n_tasks} | "
                f"Plasticity: {acc*100:.1f}% | "
                f"Rank: {rank:.1f} | "
                f"Retention: {retention*100:.1f}%"
            )
        else:
            if task_id % 10 == 0:
                print(f"  Task {task_id+1:3d}/{n_tasks} | Plasticity: {acc*100:.1f}% | Rank: {rank:.1f}")

    return {
        "label": label,
        "plasticity": plasticity_curve,
        "rank": rank_curve,
        "retention": retention_curve,
        "retention_tasks": retention_tasks,
    }


def plot_results(results: list[dict], save_dir: str = "outputs"):
    """
    Reproduce TBC Figures 1, 2, 3 side by side.
    """
    os.makedirs(save_dir, exist_ok=True)

    colors = {
        "Standard backprop": "gold",
        "Continual backprop": "#7F77DD",
        "Local plasticity (TBC)": "#1D9E75",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("TBC Post 4 Replication: Continual Learning Benchmark", fontsize=14)

    # Figure 1 — Plasticity
    ax = axes[0]
    for r in results:
        color = colors.get(r["label"], "gray")
        ax.plot(r["plasticity"], color=color, linewidth=1.5, label=r["label"], alpha=0.85)
    ax.set_xlabel("Task number")
    ax.set_ylabel("Accuracy on new task (%)")
    ax.set_title("Plasticity (TBC Fig 1)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Figure 2 — Effective rank
    ax = axes[1]
    for r in results:
        color = colors.get(r["label"], "gray")
        ax.plot(r["rank"], color=color, linewidth=1.5, label=r["label"], alpha=0.85)
    ax.set_xlabel("Task number")
    ax.set_ylabel("Effective rank")
    ax.set_title("Representation quality (TBC Fig 2)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Figure 3 — Retention
    ax = axes[2]
    for r in results:
        color = colors.get(r["label"], "gray")
        ax.plot(
            r["retention_tasks"],
            [v * 100 for v in r["retention"]],
            color=color, linewidth=1.5, label=r["label"], alpha=0.85
        )
    ax.set_xlabel("Task number")
    ax.set_ylabel("Avg accuracy on all past tasks (%)")
    ax.set_title("Retention / forgetting (TBC Fig 3)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "continual_learning_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_tasks", type=int, default=200, help="Number of sequential tasks (20 for smoke test)")
    parser.add_argument("--n_train", type=int, default=1000, help="Training samples per task")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden layer size")
    parser.add_argument("--retention_every", type=int, default=10, help="Evaluate retention every N tasks")
    parser.add_argument("--epochs_per_task", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build benchmark
    benchmark = PermutedMNIST(
        n_tasks=args.n_tasks,
        n_train_per_task=args.n_train,
    )

    # Run all three methods
    experiments = [
        ("Standard backprop", StandardNetwork(hidden_dim=args.hidden_dim)),
        ("Continual backprop", ContinualBackprop(hidden_dim=args.hidden_dim)),
        ("Local plasticity (TBC)", LocalPlasticityNet(hidden_dim=args.hidden_dim)),
    ]

    all_results = []
    for label, model in experiments:
        model = model.to(device)
        result = run_experiment(
            model, benchmark, args.n_tasks, label, device,
            eval_retention_every=args.retention_every,epochs_per_task=3
        )
        all_results.append(result)

    # Plot
    plot_results(all_results)

    # Save JSON summary
    summary = []
    for r in all_results:
        summary.append({
            "label": r["label"],
            "final_plasticity": round(r["plasticity"][-1] * 100, 2),
            "mean_plasticity": round(np.mean(r["plasticity"]) * 100, 2),
            "final_rank": round(r["rank"][-1], 2),
            "mean_rank": round(np.mean(r["rank"]), 2),
            "final_retention": round(r["retention"][-1] * 100, 2),
        })

    with open("outputs/continual_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print(f"\n{'='*65}")
    print(f"{'Method':<25} {'Plasticity':>12} {'Eff. Rank':>10} {'Retention':>10}")
    print(f"{'='*65}")
    for s in summary:
        print(f"{s['label']:<25} {s['mean_plasticity']:>11.1f}% {s['mean_rank']:>10.1f} {s['final_retention']:>9.1f}%")
    print(f"{'='*65}")
    print(f"\nTBC reported: Local plasticity ~97% plasticity, ~2× rank vs reactive")
    print(f"Outputs saved to ./outputs/")


if __name__ == "__main__":
    main()