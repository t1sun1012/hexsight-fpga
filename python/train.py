"""Train the HexSight CPU baseline and report validation results only."""

import argparse
import copy
import csv
import importlib.metadata
import json
import platform
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from hexsight.metrics import classification_metrics, inverse_frequency_weights
from hexsight.model import architecture_details, build_model
from hexsight.reporting import plot_results
from hexsight.training_data import load_training_data, sha256_file

REPO = Path(__file__).resolve().parents[1]


def evaluate(model, inputs, targets, weights, classes, batch_size=2048):
    model.eval()
    predictions = []
    weighted_sum = 0.0
    unweighted_sum = 0.0
    with torch.inference_mode():
        for start in range(0, len(targets), batch_size):
            labels = targets[start:start + batch_size]
            logits = model(inputs[start:start + batch_size])
            weighted_sum += F.cross_entropy(logits, labels, weight=weights, reduction="sum").item()
            unweighted_sum += F.cross_entropy(logits, labels, reduction="sum").item()
            predictions.append(logits.argmax(dim=1).numpy())
    predictions = np.concatenate(predictions)
    metrics = classification_metrics(targets.numpy(), predictions, classes)
    metrics["weighted_loss"] = weighted_sum / weights[targets].double().sum().item()
    metrics["unweighted_loss"] = unweighted_sum / len(targets)
    return metrics, predictions


def smoke_subset(labels, count_per_class, num_classes):
    return np.concatenate([np.flatnonzero(labels == i)[:count_per_class] for i in range(num_classes)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--audit-dir", type=Path, default=REPO / "results/data_audit")
    parser.add_argument("--config", type=Path, default=REPO / "configs/cnn.json")
    parser.add_argument("--run-dir", type=Path, required=True, help="New directory for this experiment")
    parser.add_argument("--epochs", type=int, help="Override the configured maximum epochs")
    parser.add_argument("--threads", type=int, default=1, help="CPU threads; recorded for reproducibility")
    parser.add_argument("--smoke", action="store_true", help="3 epochs on 256 train / 64 validation examples per class")
    args = parser.parse_args()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error("Run directory is not empty. Use a new directory to preserve previous results.")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["training"]["device"] != "cpu":
        parser.error("This baseline runner supports CPU training only")
    if config["training"]["class_weighting"] != "inverse_training_frequency":
        parser.error("Unsupported class weighting")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.data_dir is None:
        settings_file = REPO / "local.settings.json"
        local = json.loads(settings_file.read_text()) if settings_file.exists() else {}
        args.data_dir = Path(local.get("data_dir", REPO / "data"))
    max_epochs = args.epochs if args.epochs is not None else (3 if args.smoke else config["training"]["max_epochs"])
    if max_epochs < 1:
        parser.error("--epochs must be positive")
    config = copy.deepcopy(config)
    config["training"]["max_epochs"] = max_epochs
    seed, classes = config["seed"], config["classes"]
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    total_start = time.perf_counter()
    data = load_training_data(args.data_dir, args.audit_dir, config)
    if args.smoke:
        for prefix, per_class in (("train", 256), ("val", 64)):
            selected = smoke_subset(data[f"{prefix}_labels"], per_class, len(classes))
            for kind in ("pixels", "labels", "indices"):
                data[f"{prefix}_{kind}"] = data[f"{prefix}_{kind}"][selected]
    divisor = config["preprocessing"]["normalization_divisor"]
    train_inputs = torch.from_numpy(data["train_pixels"].astype(np.float32) / divisor)
    val_inputs = torch.from_numpy(data["val_pixels"].astype(np.float32) / divisor)
    train_labels = torch.from_numpy(data["train_labels"])
    val_labels = torch.from_numpy(data["val_labels"])
    weights = inverse_frequency_weights(train_labels, len(classes))
    model = build_model(config)
    architecture = architecture_details(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], foreach=False)
    generator = torch.Generator().manual_seed(seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "architecture": architecture,
        "config": config, "smoke": args.smoke, "threads": args.threads, "device": "cpu",
        "train_count": len(train_labels), "validation_count": len(val_labels),
        "class_weights": weights.tolist(), "selection_metric": "minimum validation weighted cross-entropy",
        "cache_key": data["cache_key"], "cache_sha256": data["cache_sha256"],
        "split_sha256": data["split_sha256"], "source_sha256": data["source_sha256"],
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy", "pillow", "matplotlib")},
        "python": sys.version.split()[0], "platform": platform.platform(), "processor": platform.processor(),
        "code_sha256": {str(path.relative_to(REPO)).replace("\\", "/"): sha256_file(path)
                        for path in sorted((REPO / "python").rglob("*.py")) if "tests" not in path.parts},
        "test_evaluated": False,
    }
    (args.run_dir / "run.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    initial, _ = evaluate(model, val_inputs, val_labels, weights, classes)
    print(f"Initial validation accuracy: {initial['accuracy']:.2%}; loss: {initial['weighted_loss']:.4f}", flush=True)
    print(f"Training {len(train_labels):,} images; validating {len(val_labels):,}; parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    best_loss, best_epoch, stale = float("inf"), 0, 0
    history = []
    training_start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        order = torch.randperm(len(train_labels), generator=generator)
        numerator = 0.0
        denominator = 0.0
        correct = 0
        batch_size = config["training"]["batch_size"]
        for start in range(0, len(order), batch_size):
            selected = order[start:start + batch_size]
            inputs, labels = train_inputs[selected], train_labels[selected]
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = F.cross_entropy(logits, labels, weight=weights)
            if not torch.isfinite(loss):
                raise RuntimeError("Training produced a non-finite loss")
            loss.backward()
            optimizer.step()
            batch_weight = weights[labels].sum().item()
            numerator += loss.item() * batch_weight
            denominator += batch_weight
            correct += int((logits.detach().argmax(dim=1) == labels).sum())
        validation, _ = evaluate(model, val_inputs, val_labels, weights, classes)
        if not np.isfinite(validation["weighted_loss"]):
            raise RuntimeError("Validation produced a non-finite loss")
        improved = validation["weighted_loss"] < best_loss
        if improved:
            best_loss, best_epoch, stale = validation["weighted_loss"], epoch, 0
            torch.save({
                "state_dict": model.state_dict(), "config": config, "epoch": epoch,
                "class_weights": weights.tolist(), "validation_metrics": validation,
                "split_sha256": data["split_sha256"], "cache_key": data["cache_key"],
            }, args.run_dir / "best.pt")
        else:
            stale += 1
        row = {
            "epoch": epoch, "train_weighted_loss": numerator / denominator,
            "train_online_accuracy": correct / len(train_labels),
            "val_weighted_loss": validation["weighted_loss"], "val_accuracy": validation["accuracy"],
            "val_macro_recall": validation["macro_recall"], "seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        with (args.run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(f"Epoch {epoch:02d} | train loss {row['train_weighted_loss']:.4f} | val loss {row['val_weighted_loss']:.4f} | accuracy {row['val_accuracy']:.2%} | macro recall {row['val_macro_recall']:.2%} | {row['seconds']:.1f}s" + (" * saved" if improved else ""), flush=True)
        if stale >= config["training"]["early_stopping_patience"]:
            print("Early stopping: validation loss has not improved.", flush=True)
            break
    training_seconds = time.perf_counter() - training_start
    checkpoint = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    metrics, predictions = evaluate(model, val_inputs, val_labels, weights, classes)
    if metrics != checkpoint["validation_metrics"]:
        raise RuntimeError("Reloaded checkpoint does not reproduce its saved validation metrics")
    if args.smoke and metrics["weighted_loss"] >= initial["weighted_loss"]:
        raise RuntimeError("Smoke run did not reduce validation loss")
    metrics.update({
        "architecture": architecture,
        "best_epoch": best_epoch, "epochs_completed": len(history), "initial_validation": initial,
        "parameters": sum(p.numel() for p in model.parameters()),
        "weights": sum(p.numel() for name, p in model.named_parameters() if name.endswith("weight")),
        "training_seconds": training_seconds, "total_seconds_before_plots": time.perf_counter() - total_start,
        "checkpoint_sha256": sha256_file(args.run_dir / "best.pt"), "test_evaluated": False,
        "smoke": args.smoke,
    })
    (args.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (args.run_dir / "per_class.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics["per_class"][0]))
        writer.writeheader()
        writer.writerows(metrics["per_class"])
    np.savez_compressed(args.run_dir / "validation_predictions.npz",
                        source_indices=data["val_indices"], targets=data["val_labels"], predictions=predictions)
    plot_results(history, metrics, data["val_pixels"], predictions, data["val_labels"], classes, best_epoch, args.run_dir, model_name=architecture["type"].upper())
    print(f"Best epoch {best_epoch}: accuracy {metrics['accuracy']:.2%}, macro recall {metrics['macro_recall']:.2%}", flush=True)
    print(f"Checkpoint reload verified. Saved results to {args.run_dir}", flush=True)


if __name__ == "__main__":
    main()
