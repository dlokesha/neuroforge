"""
db.py — Supabase client for logging all experiment data.

Stores:
  - runs         → accuracy curves, final numbers per experiment
  - ablation     → whole/center/periphery breakdown
  - spike_cache  → bio-preprocessed vectors (avoid recomputing)
  - checkpoints  → references to saved HF model files
"""

import io
import os

import numpy as np
from supabase import create_client, Client


def get_client() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    return create_client(url, key)


def log_run(
    n_samples: int,
    epochs: int,
    baseline_curve: list,
    bio_curve: list,
) -> str:
    """Insert a completed run and return its UUID."""
    client = get_client()
    payload = {
        "n_samples": n_samples,
        "epochs": epochs,
        "baseline_final_acc": round(baseline_curve[-1], 6),
        "bio_final_acc": round(bio_curve[-1], 6),
        "improvement": round(bio_curve[-1] - baseline_curve[-1], 6),
        "baseline_curve": baseline_curve,
        "bio_curve": bio_curve,
    }
    response = client.table("runs").insert(payload).execute()
    run_id = response.data[0]["id"]
    print(f"Logged run → {run_id}")
    return run_id


def log_ablation(run_id: str, ablation_results: dict):
    """Insert ablation rows (whole/center/periphery) linked to a run."""
    client = get_client()
    rows = [
        {"run_id": run_id, "region": region, "accuracy": round(acc, 6)}
        for region, acc in ablation_results.items()
    ]
    client.table("ablation_results").insert(rows).execute()
    print(f"Logged ablation results for run {run_id}")


def cache_spike_vectors(split: str, n_samples: int, vectors: np.ndarray):
    """
    Serialize and store spike-rate vectors to Supabase.
    Avoids rerunning the slow reservoir computation on every training run.

    Args:
        split: 'train' or 'test'
        n_samples: number of samples (used as cache key)
        vectors: (N, n_units) float array
    """
    client = get_client()

    # Serialize numpy array to bytes
    buf = io.BytesIO()
    np.save(buf, vectors)
    vector_bytes = buf.getvalue()

    # Check if cache already exists for this split+size
    existing = (
        client.table("spike_cache")
        .select("id")
        .eq("split", split)
        .eq("n_samples", n_samples)
        .execute()
    )
    if existing.data:
        # Update existing
        client.table("spike_cache").update({"vectors": vector_bytes.hex()}).eq(
            "split", split
        ).eq("n_samples", n_samples).execute()
        print(f"Updated spike cache: {split} / {n_samples} samples")
    else:
        # Insert new
        client.table("spike_cache").insert(
            {"split": split, "n_samples": n_samples, "vectors": vector_bytes.hex()}
        ).execute()
        print(f"Saved spike cache: {split} / {n_samples} samples")


def load_spike_vectors(split: str, n_samples: int) -> np.ndarray | None:
    """
    Load cached spike-rate vectors from Supabase.
    Returns None if no cache exists for this split+size.
    """
    client = get_client()
    response = (
        client.table("spike_cache")
        .select("vectors")
        .eq("split", split)
        .eq("n_samples", n_samples)
        .execute()
    )
    if not response.data:
        return None

    vector_bytes = bytes.fromhex(response.data[0]["vectors"])
    buf = io.BytesIO(vector_bytes)
    vectors = np.load(buf)
    print(f"Loaded spike cache from Supabase: {split} / {n_samples} samples")
    return vectors


def log_checkpoint(run_id: str, model_type: str, hf_repo: str, final_acc: float):
    """Record that a model checkpoint was saved to HuggingFace."""
    client = get_client()
    client.table("checkpoints").insert(
        {
            "run_id": run_id,
            "model_type": model_type,
            "hf_repo": hf_repo,
            "final_acc": round(final_acc, 6),
        }
    ).execute()
    print(f"Logged checkpoint: {model_type} → {hf_repo}")


def fetch_all_runs() -> list[dict]:
    """Fetch all runs for display in the Gradio dashboard."""
    client = get_client()
    response = (
        client.table("runs")
        .select("*")
        .order("created_at", desc=True)
        .execute()
    )
    return response.data
