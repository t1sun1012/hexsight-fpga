"""Fresh-process verification of exported weights, RTL vectors, and validation logits."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hexsight.fixed_point_eval import REPO, evaluate_fixed_point, load_validation, verify_vectors
from hexsight.fixed_point_io import load_bundle, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=REPO / "results/cnn-8filters")
    parser.add_argument("--model-dir", type=Path, default=REPO / "model/cnn-fixed-point")
    parser.add_argument("--results-dir", type=Path, default=REPO / "results/fixed-point")
    parser.add_argument("--vectors-dir", type=Path, default=REPO / "test_vectors/cnn-fixed-point")
    parser.add_argument("--audit-dir", type=Path, default=REPO / "results/data_audit")
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model, manifest = load_bundle(args.model_dir)
    run = json.loads((args.results_dir / "run.json").read_text())
    expected = json.loads((args.results_dir / "metrics.json").read_text())
    for path, key in ((args.model_dir / "manifest.json", "model_manifest_sha256"),
                      (args.vectors_dir / "manifest.json", "vector_manifest_sha256"),
                      (args.results_dir / "validation_predictions.npz", "predictions_sha256")):
        if sha256(path) != run[key]:
            raise ValueError(f"Report artifact hash mismatch: {key}")
    data, floating, requantized, expected_float, baseline_run, baseline = load_validation(
        args.run_dir, args.audit_dir, args.data_dir)
    provenance = manifest["provenance"]
    if (provenance != run["provenance"] or provenance["checkpoint_sha256"] != baseline["checkpoint_sha256"]
            or provenance["config"] != baseline_run["config"]
            or provenance["baseline_run_sha256"] != sha256(args.run_dir / "run.json")):
        raise ValueError("Fixed-point and baseline provenance mismatch")
    for key in ("cache_key", "cache_sha256", "split_sha256", "source_sha256"):
        if provenance[key] != data[key]:
            raise ValueError(f"Fixed-point data provenance mismatch: {key}")
    for name in model.parameters:
        np.testing.assert_array_equal(model.parameters[name], requantized.parameters[name])
    vector_check = verify_vectors(args.vectors_dir, model, args.model_dir)
    if run["validation_count"] != len(data["val_pixels"]) or run["vector_count"] != vector_check["cases_checked"]:
        raise ValueError("Run counts disagree with validation data or RTL vectors")
    metrics, predictions = evaluate_fixed_point(model, floating, data["val_pixels"], data["val_labels"], expected_float)
    with np.load(args.results_dir / "validation_predictions.npz", allow_pickle=False) as saved:
        for name, values in (("source_indices", data["val_indices"]), ("targets", data["val_labels"]),
                             ("predictions", predictions), ("float_predictions", expected_float)):
            np.testing.assert_array_equal(saved[name], values)
    for key in metrics:
        if key != "evaluation_seconds" and metrics[key] != expected[key]:
            raise ValueError(f"Recomputed metric differs: {key}")
    result = {
        "model_manifest_sha256": run["model_manifest_sha256"],
        "validation_predictions_checked": len(predictions), "prediction_mismatches": 0,
        "integer_logits_checked": int(len(predictions) * 16),
        "logits_sha256_little_endian_int64": metrics["logits_sha256_little_endian_int64"],
        "source_parameter_requantization_mismatches": 0, "vectors": vector_check,
        "accuracy": metrics["accuracy"], "test_evaluated": False,
    }
    write_json(args.results_dir / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
