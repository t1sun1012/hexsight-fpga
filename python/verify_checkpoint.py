"""Reload a completed run and independently reproduce its validation predictions."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hexsight.model import build_model
from hexsight.training_data import load_training_data, sha256_file
from train import evaluate, smoke_subset

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--audit-dir", type=Path, default=REPO / "results/data_audit")
    args = parser.parse_args()
    run = json.loads((args.run_dir / "run.json").read_text())
    expected = json.loads((args.run_dir / "metrics.json").read_text())
    checkpoint_path = args.run_dir / "best.pt"
    if sha256_file(checkpoint_path) != expected["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash differs from the saved report")
    config = run["config"]
    if args.data_dir is None:
        settings_file = REPO / "local.settings.json"
        settings = json.loads(settings_file.read_text()) if settings_file.exists() else {}
        args.data_dir = Path(settings.get("data_dir", REPO / "data"))
    torch.set_num_threads(run["threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    data = load_training_data(args.data_dir, args.audit_dir, config)
    if data["cache_key"] != run["cache_key"] or data["cache_sha256"] != run["cache_sha256"]:
        raise ValueError("Preprocessing cache differs from the trained run")
    pixels, labels, indices = data["val_pixels"], data["val_labels"], data["val_indices"]
    if run["smoke"]:
        selected = smoke_subset(labels, 64, len(config["classes"]))
        pixels, labels, indices = pixels[selected], labels[selected], indices[selected]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["config"] != config or checkpoint["split_sha256"] != run["split_sha256"]:
        raise ValueError("Checkpoint configuration/split differs from the run manifest")
    model = build_model(config)
    model.load_state_dict(checkpoint["state_dict"])
    inputs = torch.from_numpy(pixels.astype(np.float32) / config["preprocessing"]["normalization_divisor"])
    metrics, predictions = evaluate(model, inputs, torch.from_numpy(labels),
                                    torch.tensor(checkpoint["class_weights"]), config["classes"])
    with np.load(args.run_dir / "validation_predictions.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(indices, saved["source_indices"])
        np.testing.assert_array_equal(labels, saved["targets"])
        np.testing.assert_array_equal(predictions, saved["predictions"])
    np.testing.assert_array_equal(metrics["confusion_matrix"], expected["confusion_matrix"])
    for key in ("accuracy", "macro_recall", "macro_f1", "weighted_loss", "unweighted_loss"):
        if not np.isclose(metrics[key], expected[key], rtol=1e-7, atol=1e-9):
            raise ValueError(f"Saved and recomputed metric differ: {key}")
    result = {
        "checkpoint_sha256": expected["checkpoint_sha256"],
        "validation_predictions_checked": len(predictions), "mismatches": 0,
        "accuracy": metrics["accuracy"], "macro_recall": metrics["macro_recall"],
        "test_evaluated": False,
    }
    (args.run_dir / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
