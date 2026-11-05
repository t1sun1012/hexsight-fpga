"""Validation and deterministic RTL vectors for the fixed-point CNN."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .fixed_point import ACT_MAX, REQUANT_SHIFT, FixedPointCNN, round_shift, scalar_forward
from .fixed_point_io import read_mem, sha256, write_json, write_mem
from .metrics import classification_metrics
from .model import build_model
from .training_data import load_training_data

REPO = Path(__file__).resolve().parents[2]
STAGES = {
    "pixels": ((256,), 8, False),
    "conv_acc": ((8, 14, 14), 40, True),
    "relu": ((8, 14, 14), 16, True),
    "pool": ((8, 7, 7), 16, True),
    "logits": ((16,), 40, True),
    "predictions": ((), 4, False),
}


def load_validation(run_dir, audit_dir, data_dir=None):
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    config = run["config"]
    required_model = {"type": "cnn", "filters": 8, "kernel_size": 3, "pool_size": 2, "activation": "relu"}
    if (run["smoke"] or config["model"] != required_model or config["classes"] != "0123456789ABCDEF"
            or config["preprocessing"]["image_size"] != 16
            or config["preprocessing"]["normalization_divisor"] != 256):
        raise ValueError("This hardware contract requires the complete 16x16, eight-filter CNN baseline")
    checkpoint_path = run_dir / "best.pt"
    if sha256(checkpoint_path) != metrics["checkpoint_sha256"]:
        raise ValueError("Baseline checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (checkpoint["config"] != config or checkpoint["split_sha256"] != run["split_sha256"]
            or checkpoint["cache_key"] != run["cache_key"] or checkpoint["epoch"] != metrics["best_epoch"]):
        raise ValueError("Checkpoint configuration/data identity mismatch")
    if data_dir is None:
        settings_path = REPO / "local.settings.json"
        settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
        data_dir = Path(settings.get("data_dir", REPO / "data"))
    data = load_training_data(Path(data_dir), Path(audit_dir), config)
    for key in ("cache_key", "cache_sha256", "split_sha256"):
        if data[key] != run[key]:
            raise ValueError(f"Baseline and current data differ: {key}")
    with np.load(run_dir / "validation_predictions.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["source_indices"], data["val_indices"])
        np.testing.assert_array_equal(saved["targets"], data["val_labels"])
        predictions = saved["predictions"].copy()
    floating = build_model(config)
    floating.load_state_dict(checkpoint["state_dict"])
    floating.eval()
    integer = FixedPointCNN.from_state_dict(checkpoint["state_dict"])
    return data, floating, integer, predictions, run, metrics


def evaluate_fixed_point(integer, floating, pixels, targets, expected_float_predictions, batch_size=128):
    start = time.perf_counter()
    all_predictions, all_float_predictions = [], []
    ranges = {name: {"min": None, "max": None} for name in ("conv_acc", "relu", "pool", "logits")}
    saturation_count = 0
    error_sum, error_max, score_count = 0.0, 0.0, 0
    logit_digest = hashlib.sha256()
    with torch.inference_mode():
        for offset in range(0, len(pixels), batch_size):
            batch = pixels[offset:offset + batch_size]
            trace = integer.forward(batch)
            float_logits = floating(torch.from_numpy(batch.astype(np.float32) / 256)).numpy()
            float_predictions = float_logits.argmax(axis=1)
            np.testing.assert_array_equal(float_predictions, expected_float_predictions[offset:offset + len(batch)])
            all_predictions.append(trace["predictions"])
            all_float_predictions.append(float_predictions)
            logit_digest.update(trace["logits"].astype("<i8").tobytes())
            errors = np.abs(trace["logits"].astype(np.float64) / (1 << 20) - float_logits)
            error_sum += float(errors.sum())
            error_max = max(error_max, float(errors.max()))
            score_count += errors.size
            saturation_count += int(np.count_nonzero(round_shift(trace["conv_acc"], REQUANT_SHIFT) > ACT_MAX))
            for name, bounds in ranges.items():
                low, high = int(trace[name].min()), int(trace[name].max())
                bounds["min"] = low if bounds["min"] is None else min(bounds["min"], low)
                bounds["max"] = high if bounds["max"] is None else max(bounds["max"], high)
            if offset % (batch_size * 64) == 0:
                print(f"Checked integer/float inference: {min(offset + len(batch), len(pixels)):,}/{len(pixels):,}", flush=True)
    predictions = np.concatenate(all_predictions)
    float_predictions = np.concatenate(all_float_predictions)
    metrics = classification_metrics(targets, predictions, "0123456789ABCDEF")
    floating_metrics = classification_metrics(targets, float_predictions, "0123456789ABCDEF")
    metrics.update({
        "float_accuracy": floating_metrics["accuracy"],
        "float_macro_recall": floating_metrics["macro_recall"],
        "accuracy_change_percentage_points": 100 * (metrics["accuracy"] - floating_metrics["accuracy"]),
        "macro_recall_change_percentage_points": 100 * (metrics["macro_recall"] - floating_metrics["macro_recall"]),
        "float_prediction_disagreements": int(np.count_nonzero(predictions != float_predictions)),
        "float_prediction_agreement": float(np.mean(predictions == float_predictions)),
        "float_correct_integer_wrong": int(np.count_nonzero((float_predictions == targets) & (predictions != targets))),
        "integer_correct_float_wrong": int(np.count_nonzero((predictions == targets) & (float_predictions != targets))),
        "logit_mean_absolute_error": error_sum / score_count,
        "logit_max_absolute_error": error_max,
        "integer_ranges": ranges,
        "activation_saturation_count": saturation_count,
        "convolution_activations_checked": int(len(pixels) * 8 * 14 * 14),
        "logits_sha256_little_endian_int64": logit_digest.hexdigest(),
        "evaluation_seconds": time.perf_counter() - start,
        "timing_note": "Combined integer/float validation and diagnostics; not an FPGA or CPU inference benchmark.",
        "test_evaluated": False,
    })
    return metrics, predictions


def make_vectors(directory, model, model_dir, data):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    images, cases = [], []
    for label, character in enumerate("0123456789ABCDEF"):
        index = int(np.flatnonzero(data["val_labels"] == label)[0])
        images.append(data["val_pixels"][index])
        cases.append({"name": f"validation_{character}", "source": "official training split / validation partition",
                      "source_index": int(data["val_indices"][index]), "target": label})
    rows, cols = np.indices((16, 16))
    synthetic = {
        "blank": np.zeros((16, 16), dtype=np.uint8),
        "full_white": np.full((16, 16), 255, dtype=np.uint8),
        "checkerboard": ((rows + cols) % 2 * 255).astype(np.uint8),
        "impulse_top_left": np.pad(np.array([[255]], dtype=np.uint8), ((0, 15), (0, 15))),
        "impulse_bottom_right": np.pad(np.array([[255]], dtype=np.uint8), ((15, 0), (15, 0))),
        "horizontal_ramp": (cols * 17).astype(np.uint8),
        "vertical_ramp": (rows * 17).astype(np.uint8),
        "random_seed_311": np.random.default_rng(311).integers(0, 256, (16, 16), dtype=np.uint8),
    }
    for name, image in synthetic.items():
        images.append(image.reshape(256))
        cases.append({"name": name, "source": "synthetic arithmetic input", "source_index": None, "target": None})
    pixels = np.stack(images)
    trace, scalar = model.forward(pixels), scalar_forward(model, pixels)
    for key in trace:
        np.testing.assert_array_equal(trace[key], scalar[key], err_msg=key)
    files = {}
    for name, (shape, bits, signed) in STAGES.items():
        values = pixels if name == "pixels" else trace[name]
        filename = name + ".mem"
        write_mem(directory / filename, values, bits, signed)
        np.testing.assert_array_equal(read_mem(directory / filename, bits, values.shape, signed), values)
        files[name] = {"file": filename, "bits": bits, "signed": signed,
                       "shape_per_case": list(shape), "words_per_case": int(np.prod(shape)),
                       "sha256": sha256(directory / filename)}
    for index, case in enumerate(cases):
        case.update({"case_index": index, "expected_prediction": int(trace["predictions"][index])})
    manifest = {
        "version": 1, "model_manifest_sha256": sha256(Path(model_dir) / "manifest.json"),
        "cases": cases, "files": files,
        "layout": "Cases concatenated; each case flattened in C order. Offset = case_index * words_per_case.",
        "scalar_cross_checks": len(cases), "test_evaluated": False,
    }
    write_json(directory / "manifest.json", manifest)
    return manifest


def verify_vectors(directory, model, model_dir):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["version"] != 1 or manifest["model_manifest_sha256"] != sha256(Path(model_dir) / "manifest.json"):
        raise ValueError("Vector/model manifest mismatch")
    if set(manifest["files"]) != set(STAGES) or not manifest["cases"]:
        raise ValueError("Invalid vector stages/cases")
    count = len(manifest["cases"])
    values = {}
    for name, (shape, bits, signed) in STAGES.items():
        spec = manifest["files"][name]
        expected = {"file": name + ".mem", "bits": bits, "signed": signed,
                    "shape_per_case": list(shape), "words_per_case": int(np.prod(shape))}
        if any(spec.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Invalid vector layout: {name}")
        if sha256(directory / (name + ".mem")) != spec["sha256"]:
            raise ValueError(f"Vector hash mismatch: {name}")
        values[name] = read_mem(directory / (name + ".mem"), bits, (count, *shape), signed)
    pixels = values.pop("pixels").astype(np.uint8)
    vectorized, scalar = model.forward(pixels), scalar_forward(model, pixels)
    for name, expected in values.items():
        np.testing.assert_array_equal(vectorized[name], expected, err_msg=name)
        np.testing.assert_array_equal(scalar[name], expected, err_msg=name)
    for index, case in enumerate(manifest["cases"]):
        if case["case_index"] != index or case["expected_prediction"] != int(values["predictions"][index]):
            raise ValueError("Vector case metadata differs from its memory file")
    return {"cases_checked": count, "intermediate_words_checked": sum(v.size for v in values.values()),
            "vectorized_scalar_and_memory_mismatches": 0}
