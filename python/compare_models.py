"""Compare two verified full validation runs on exactly the same examples."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hexsight.model import architecture_details, build_model
from hexsight.training_data import sha256_file


def load_run(path):
    run = json.loads((path / "run.json").read_text())
    metrics = json.loads((path / "metrics.json").read_text())
    verified = json.loads((path / "verification.json").read_text())
    if run["smoke"] or metrics["smoke"]:
        raise ValueError("Comparison requires full runs, not smoke subsets")
    digest = sha256_file(path / "best.pt")
    if digest != metrics["checkpoint_sha256"] or digest != verified["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash differs from the verified result")
    if verified["mismatches"] or verified["validation_predictions_checked"] != metrics["count"]:
        raise ValueError("A complete successful checkpoint verification is required")
    if run["test_evaluated"] or metrics["test_evaluated"] or verified["test_evaluated"]:
        raise ValueError("This comparison is for validation-only experiments")
    with np.load(path / "validation_predictions.npz", allow_pickle=False) as stored:
        predictions = {name: stored[name] for name in ("source_indices", "targets", "predictions")}
    return run, metrics, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlp-dir", type=Path, default=Path("results/baseline-256-32-16"))
    parser.add_argument("--cnn-dir", type=Path, default=Path("results/cnn-8filters"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/model-comparison"))
    args = parser.parse_args()
    old_run, old, old_predictions = load_run(args.mlp_dir)
    new_run, new, new_predictions = load_run(args.cnn_dir)
    for field in ("classes", "dataset_split", "seed", "validation_fraction", "preprocessing", "training"):
        if old_run["config"][field] != new_run["config"][field]:
            raise ValueError(f"Training/evaluation control changed: {field}")
    for field in ("cache_sha256", "split_sha256", "source_sha256", "train_count",
                  "validation_count", "class_weights", "selection_metric", "threads", "versions"):
        if old_run[field] != new_run[field]:
            raise ValueError(f"Run provenance differs: {field}")
    for name in ("source_indices", "targets"):
        np.testing.assert_array_equal(old_predictions[name], new_predictions[name])
    old_correct = old_predictions["predictions"] == old_predictions["targets"]
    new_correct = new_predictions["predictions"] == new_predictions["targets"]
    old_arch = architecture_details(build_model(old_run["config"]))
    new_arch = architecture_details(build_model(new_run["config"]))
    if old_arch["type"] != "mlp" or new_arch["type"] != "cnn":
        raise ValueError("Expected an MLP reference and a CNN candidate")
    summary = {
        "comparison": "Single-seed validation comparison; no FPGA latency or final test evaluation.",
        "validation_count": len(old_correct), "seed": old_run["config"]["seed"],
        "controls_verified": True, "test_evaluated": False,
        "accuracy_gain_percentage_points": 100 * (new["accuracy"] - old["accuracy"]),
        "macro_recall_gain_percentage_points": 100 * (new["macro_recall"] - old["macro_recall"]),
        "mlp_only_correct": int(np.count_nonzero(old_correct & ~new_correct)),
        "cnn_only_correct": int(np.count_nonzero(~old_correct & new_correct)),
        "both_correct": int(np.count_nonzero(old_correct & new_correct)),
        "both_wrong": int(np.count_nonzero(~old_correct & ~new_correct)),
        "models": {},
    }
    for name, metrics, architecture in (("mlp", old, old_arch), ("cnn", new, new_arch)):
        summary["models"][name] = {
            **{key: metrics[key] for key in ("accuracy", "macro_recall", "macro_f1", "weighted_loss",
                                            "best_epoch", "epochs_completed", "training_seconds", "checkpoint_sha256")},
            "architecture": architecture,
        }
    classes = old_run["config"]["classes"]
    rows = []
    for i, character in enumerate(classes):
        a, b = old["per_class"][i], new["per_class"][i]
        rows.append({
            "character": character, "support": a["support"],
            "mlp_recall": a["recall"], "cnn_recall": b["recall"],
            "recall_gain_percentage_points": 100 * (b["recall"] - a["recall"]),
            "mlp_precision": a["precision"], "cnn_precision": b["precision"],
        })
    summary["key_confusions"] = []
    for actual, predicted in (("8", "B"), ("0", "D")):
        i, j = classes.index(actual), classes.index(predicted)
        summary["key_confusions"].append({
            "true": actual, "predicted": predicted,
            "mlp_count": old["confusion_matrix"][i][j], "cnn_count": new["confusion_matrix"][i][j],
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "per_class_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout="constrained")
    x = np.arange(len(classes))
    for axis, metric in zip(axes, ("recall", "precision")):
        axis.bar(x - 0.2, [100 * row[f"mlp_{metric}"] for row in rows], width=0.4, label="MLP", color="#94a3b8")
        axis.bar(x + 0.2, [100 * row[f"cnn_{metric}"] for row in rows], width=0.4, label="CNN", color="#0f766e")
        axis.set(xticks=x, xticklabels=list(classes), ylim=(0, 105), ylabel=f"{metric.title()} (%)")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(loc="lower right")
    axes[1].set_xlabel("Character")
    fig.suptitle("HexSight | same 57,612 validation images, same training settings")
    fig.savefig(args.output_dir / "per_class_comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
