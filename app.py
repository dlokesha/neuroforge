"""
app.py — Gradio interface for the TBC pipeline replication.
This is the entry point for the HuggingFace Space.

Tabs:
  1. Run experiment  — configure + launch training, live accuracy plot
  2. Results history — all past runs pulled from Supabase
  3. Activation spread — visualize how neural activity spreads
"""

import json
import os
import time

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch

from db import fetch_all_runs, log_ablation, log_checkpoint, log_run
from models import BaselineCNN, BioCNN
from reservoir import BioPreprocessor, MEAEncoder
from train import (
    encode_to_grid,
    load_mnist,
    run_ablation,
    train_model,
)
from torch.utils.data import DataLoader, TensorDataset


# ── Helpers ─────────────────────────────────────────────────────────────────

def build_accuracy_plot(baseline_curve, bio_curve, title="Training accuracy"):
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(baseline_curve) + 1)
    ax.plot(epochs, [a * 100 for a in baseline_curve], color="gold", linewidth=2, label="Baseline (raw MNIST)")
    ax.plot(epochs, [a * 100 for a in bio_curve], color="#1D9E75", linewidth=2, label="Bio-preprocessed")
    ax.axhline(y=10, color="gray", linestyle="--", alpha=0.5, label="Chance (10%)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def build_spread_plot(raw_image, electrode_grid, spike_readout):
    preprocessor = BioPreprocessor(n_reservoir_units=1024)
    spatial = preprocessor.reservoir.get_spatial_readout(spike_readout, grid_size=64)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Neural activity spread beyond stimulation region", fontsize=13)

    axes[0].imshow(raw_image, cmap="gray")
    axes[0].set_title("Original digit")
    axes[0].axis("off")

    axes[1].imshow(electrode_grid, cmap="gray")
    axes[1].set_title("Electrode grid")
    axes[1].axis("off")
    rect = plt.Rectangle((18, 18), 28, 28, linewidth=2, edgecolor="red", facecolor="none")
    axes[1].add_patch(rect)

    axes[2].imshow(spatial, cmap="hot")
    axes[2].set_title("Reservoir spread")
    axes[2].axis("off")
    rect2 = plt.Rectangle((18, 18), 28, 28, linewidth=2, edgecolor="cyan", facecolor="none")
    axes[2].add_patch(rect2)

    plt.tight_layout()
    return fig


# ── Tab 1: Run experiment ────────────────────────────────────────────────────

def run_experiment(n_samples, epochs, run_ablation_flag, progress=gr.Progress()):
    """Full pipeline: load → bio process → train both → log to Supabase."""
    n_samples = int(n_samples)
    epochs = int(epochs)
    logs = []

    def log(msg):
        logs.append(msg)
        return "\n".join(logs)

    progress(0, desc="Loading MNIST...")
    yield log("Loading MNIST..."), None, None

    train_images, train_labels, test_images, test_labels = load_mnist(n_samples)
    train_grids = encode_to_grid(train_images)
    test_grids = encode_to_grid(test_images)
    yield log(f"Loaded {n_samples} training samples, encoded to 64×64 grids."), None, None

    # Bio preprocessing — try Supabase cache first
    from db import load_spike_vectors, cache_spike_vectors

    progress(0.15, desc="Bio preprocessing...")
    yield log("Checking Supabase cache for spike vectors..."), None, None

    bio_train = load_spike_vectors("train", n_samples)
    bio_test = load_spike_vectors("test", n_samples // 5)

    if bio_train is None:
        yield log("No cache found. Running reservoir (this takes ~2 min)..."), None, None
        preprocessor = BioPreprocessor(n_reservoir_units=1024)
        bio_train = preprocessor.process_batch(train_grids)
        bio_test = preprocessor.process_batch(test_grids)
        cache_spike_vectors("train", n_samples, bio_train)
        cache_spike_vectors("test", n_samples // 5, bio_test)
        yield log("Spike vectors computed and saved to Supabase."), None, None
    else:
        yield log("Loaded spike vectors from Supabase cache."), None, None

    # Build dataloaders
    baseline_train_ds = TensorDataset(torch.FloatTensor(train_grids).unsqueeze(1), torch.LongTensor(train_labels))
    baseline_test_ds = TensorDataset(torch.FloatTensor(test_grids).unsqueeze(1), torch.LongTensor(test_labels))
    bio_train_ds = TensorDataset(torch.FloatTensor(bio_train), torch.LongTensor(train_labels))
    bio_test_ds = TensorDataset(torch.FloatTensor(bio_test), torch.LongTensor(test_labels))

    bl_train_loader = DataLoader(baseline_train_ds, batch_size=64, shuffle=True)
    bl_test_loader = DataLoader(baseline_test_ds, batch_size=64)
    bio_train_loader = DataLoader(bio_train_ds, batch_size=64, shuffle=True)
    bio_test_loader = DataLoader(bio_test_ds, batch_size=64)

    # Train baseline
    progress(0.3, desc="Training baseline CNN...")
    yield log("Training baseline CNN on raw MNIST..."), None, None
    baseline_model = BaselineCNN()
    _, baseline_test = train_model(baseline_model, bl_train_loader, bl_test_loader, epochs, label="Baseline")
    yield log(f"Baseline done. Final acc: {baseline_test[-1]*100:.1f}%"), None, None

    # Train bio model
    progress(0.6, desc="Training bio CNN...")
    yield log("Training Bio CNN on spike-rate vectors..."), None, None
    bio_model = BioCNN()
    _, bio_test_curve = train_model(bio_model, bio_train_loader, bio_test_loader, epochs, label="Bio")
    improvement = (bio_test_curve[-1] - baseline_test[-1]) * 100
    yield log(f"Bio done. Final acc: {bio_test_curve[-1]*100:.1f}% (Δ {improvement:+.1f}%)"), None, None

    # Build plot
    plot = build_accuracy_plot(baseline_test, bio_test_curve)

    # Ablation
    ablation_results = {}
    if run_ablation_flag:
        progress(0.8, desc="Running ablation...")
        yield log("Running ablation study (whole / center / periphery)..."), plot, None
        ablation_results = run_ablation(bio_train, bio_test, train_labels, test_labels, epochs=epochs)
        for region, acc in ablation_results.items():
            yield log(f"  {region}: {acc*100:.1f}%"), plot, None

    # Log to Supabase
    progress(0.95, desc="Saving to Supabase...")
    run_id = log_run(n_samples, epochs, baseline_test, bio_test_curve)
    if ablation_results:
        log_ablation(run_id, ablation_results)

    # Save models to HF Hub
    hf_user = os.environ.get("HF_USERNAME", "unknown")
    repo = f"{hf_user}/neuroforge"
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        torch.save(baseline_model.state_dict(), "/tmp/baseline.pt")
        torch.save(bio_model.state_dict(), "/tmp/bio.pt")
        api.upload_file(path_or_fileobj="/tmp/baseline.pt", path_in_repo=f"checkpoints/{run_id}/baseline.pt", repo_id=repo)
        api.upload_file(path_or_fileobj="/tmp/bio.pt", path_in_repo=f"checkpoints/{run_id}/bio.pt", repo_id=repo)
        log_checkpoint(run_id, "baseline", repo, baseline_test[-1])
        log_checkpoint(run_id, "bio", repo, bio_test_curve[-1])
        yield log(f"Model checkpoints saved to HF: {repo}"), plot, None
    except Exception as e:
        yield log(f"(Checkpoint upload skipped: {e})"), plot, None

    # Activation spread for sample digit
    spread_plot = build_spread_plot(train_images[0], train_grids[0], bio_train[0])

    progress(1.0, desc="Done!")
    summary = (
        f"Baseline:  {baseline_test[-1]*100:.1f}%\n"
        f"Bio:       {bio_test_curve[-1]*100:.1f}%\n"
        f"Δ improve: {improvement:+.1f}%\n"
        f"(TBC paper: +4.7%)\n"
        f"Run ID: {run_id}"
    )
    yield log("Done! " + summary), plot, spread_plot


# ── Tab 2: Results history ───────────────────────────────────────────────────

def load_history():
    try:
        runs = fetch_all_runs()
        if not runs:
            return "No runs yet.", None
        rows = []
        for r in runs:
            rows.append([
                r["created_at"][:19],
                r["n_samples"],
                r["epochs"],
                f"{r['baseline_final_acc']*100:.1f}%",
                f"{r['bio_final_acc']*100:.1f}%",
                f"{r['improvement']*100:+.1f}%",
            ])

        # Plot all bio curves
        fig, ax = plt.subplots(figsize=(8, 4))
        for r in runs[:5]:  # last 5 runs
            if r.get("bio_curve"):
                ax.plot([a * 100 for a in r["bio_curve"]], alpha=0.7, label=f"{r['created_at'][:10]} ({r['n_samples']} samples)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Bio accuracy (%)")
        ax.set_title("Bio CNN accuracy across runs")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return rows, fig
    except Exception as e:
        return f"Error fetching runs: {e}", None


# ── Tab 3: Spread visualizer ─────────────────────────────────────────────────

def visualize_spread(digit_class):
    train_images, train_labels, _, _ = load_mnist(n_samples=500)
    train_grids = encode_to_grid(train_images)

    # Find first sample of requested class
    idx = next((i for i, l in enumerate(train_labels) if l == digit_class), 0)
    preprocessor = BioPreprocessor(n_reservoir_units=1024)
    spike_readout = preprocessor.process(train_grids[idx])
    return build_spread_plot(train_images[idx], train_grids[idx], spike_readout)


# ── Build Gradio UI ──────────────────────────────────────────────────────────

with gr.Blocks(title="Neuroforge — TBC Pipeline Replication", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # TBC Biological Computing Pipeline — Replication
    Software simulation of [The Biological Computing Co.](https://www.tbc.co) MEA pipeline.

    **Real TBC:** image → living neurons → spike readout → CNN  
    **This:** image → Echo State Network → spike readout → CNN
    """)

    with gr.Tab("Run experiment"):
        with gr.Row():
            n_samples_slider = gr.Slider(100, 3000, value=500, step=100, label="Training samples")
            epochs_slider = gr.Slider(3, 20, value=10, step=1, label="Epochs")
            ablation_check = gr.Checkbox(value=True, label="Run ablation study")
        run_btn = gr.Button("Run experiment", variant="primary")

        with gr.Row():
            log_box = gr.Textbox(label="Live log", lines=12, max_lines=20)
        with gr.Row():
            acc_plot = gr.Plot(label="Accuracy curves")
            spread_plot_out = gr.Plot(label="Activation spread")

        run_btn.click(
            fn=run_experiment,
            inputs=[n_samples_slider, epochs_slider, ablation_check],
            outputs=[log_box, acc_plot, spread_plot_out],
        )

    with gr.Tab("Results history"):
        refresh_btn = gr.Button("Load from Supabase")
        history_table = gr.Dataframe(
            headers=["Date", "Samples", "Epochs", "Baseline acc", "Bio acc", "Improvement"],
            label="All runs",
        )
        history_plot = gr.Plot(label="Bio accuracy across runs")
        refresh_btn.click(fn=load_history, outputs=[history_table, history_plot])

    with gr.Tab("Activation spread"):
        gr.Markdown("Visualize how neural activity spreads beyond the stimulated region for each digit class.")
        digit_selector = gr.Slider(0, 9, value=5, step=1, label="Digit class")
        spread_btn = gr.Button("Visualize")
        spread_out = gr.Plot()
        spread_btn.click(fn=visualize_spread, inputs=digit_selector, outputs=spread_out)


if __name__ == "__main__":
    demo.launch()
