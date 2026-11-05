"""Export the Step 2 CNN as integers and evaluate it without retraining."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from hexsight.fixed_point_eval import REPO, evaluate_fixed_point, load_validation, make_vectors
from hexsight.fixed_point_io import export_bundle, load_bundle, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=REPO / "results/cnn-8filters")
    parser.add_argument("--model-dir", type=Path, default=REPO / "model/cnn-fixed-point")
    parser.add_argument("--output-dir", type=Path, default=REPO / "results/fixed-point")
    parser.add_argument("--vectors-dir", type=Path, default=REPO / "test_vectors/cnn-fixed-point")
    parser.add_argument("--audit-dir", type=Path, default=REPO / "results/data_audit")
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    destinations = [args.model_dir, args.output_dir, args.vectors_dir]
    resolved = [p.resolve() for p in destinations]
    if any(a == b or a in b.parents or b in a.parents for i, a in enumerate(resolved) for b in resolved[i + 1:]):
        raise ValueError("Use distinct, non-nested model, report, and vector directories")
    for path in destinations:
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise ValueError(f"Use a new or empty output directory: {path}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    data, floating, integer, expected, run, baseline = load_validation(args.run_dir, args.audit_dir, args.data_dir)
    for path in destinations:
        path.mkdir(parents=True, exist_ok=True)
    source_files = [*sorted((REPO / "python/hexsight").glob("*.py")),
                    REPO / "python/quantize.py", REPO / "python/verify_fixed_point.py"]
    provenance = {
        "checkpoint_sha256": baseline["checkpoint_sha256"],
        "baseline_run_sha256": sha256(args.run_dir / "run.json"),
        "selected_epoch": baseline["best_epoch"],
        "config": run["config"], "split_sha256": data["split_sha256"],
        "cache_key": data["cache_key"], "cache_sha256": data["cache_sha256"],
        "source_sha256": data["source_sha256"],
        "code_sha256": {p.relative_to(REPO).as_posix(): sha256(p) for p in source_files},
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
        "method": "post-training quantization; no retraining; fixed formats; no test-set access",
    }
    export_bundle(args.model_dir, integer, provenance)
    # The evaluated model is reconstructed from the actual exported FPGA files.
    exported, manifest = load_bundle(args.model_dir)
    for name in integer.parameters:
        np.testing.assert_array_equal(exported.parameters[name], integer.parameters[name])
    metrics, predictions = evaluate_fixed_point(exported, floating, data["val_pixels"], data["val_labels"], expected)
    np.savez_compressed(args.output_dir / "validation_predictions.npz",
                        source_indices=data["val_indices"], targets=data["val_labels"],
                        predictions=predictions, float_predictions=expected)
    write_json(args.output_dir / "metrics.json", metrics)
    with (args.output_dir / "per_class.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics["per_class"][0]))
        writer.writeheader()
        writer.writerows(metrics["per_class"])
    vectors = make_vectors(args.vectors_dir, exported, args.model_dir, data)
    write_json(args.output_dir / "run.json", {
        "provenance": provenance, "model_manifest_sha256": sha256(args.model_dir / "manifest.json"),
        "vector_manifest_sha256": sha256(args.vectors_dir / "manifest.json"),
        "predictions_sha256": sha256(args.output_dir / "validation_predictions.npz"),
        "validation_count": len(predictions), "vector_count": len(vectors["cases"]),
        "parameter_storage_bytes": manifest["storage_bytes"], "test_evaluated": False,
    })
    print(f"Fixed-point accuracy {metrics['accuracy']:.6%}; macro recall {metrics['macro_recall']:.6%}", flush=True)
    print(f"Float disagreements {metrics['float_prediction_disagreements']}; saturation count {metrics['activation_saturation_count']}", flush=True)
    print(f"Exported {len(vectors['cases'])} independently checked RTL cases.", flush=True)


if __name__ == "__main__":
    main()
