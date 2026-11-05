"""Download EMNIST, audit the hexadecimal subset, and freeze training indices.

This step does not train or evaluate a model. Only training images are previewed.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from torchvision.datasets import EMNIST

from hexsight.data import class_mapping, preprocess_image, stratified_split

REPO = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_preview(dataset, train_indices, mapping, classes, settings, path):
    # Each group shows stored pixels, upright pixels, and the model's input.
    samples = 4
    row_height = 70
    sheet = Image.new("RGB", (72 + samples * 190, 90 + len(classes) * row_height), "#f5f7fa")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), "HexSight: training samples only", fill="#10243a", font_size=20)
    draw.text((12, 38), "Per group: raw storage | upright 28x28 | model 16x16", fill="#10243a", font_size=16)
    targets = dataset.targets.numpy()
    for source_id, label in mapping.items():
        candidates = train_indices[targets[train_indices] == source_id][:samples]
        y = 80 + label * row_height
        draw.text((16, y + 16), classes[label], fill="#10243a", font_size=25)
        for column, index in enumerate(candidates):
            raw = dataset.data[int(index)].numpy()
            upright = raw.T if settings["emnist_transpose"] else raw
            processed = preprocess_image(raw, settings, transpose=settings["emnist_transpose"])
            for offset, pixels in enumerate((raw, upright, processed)):
                tile = Image.fromarray(pixels).resize((56, 56), Image.Resampling.NEAREST)
                sheet.paste(tile, (72 + column * 190 + offset * 60, y))
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO / "data")
    parser.add_argument("--output-dir", type=Path, default=REPO / "results" / "data_audit")
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "baseline.json")
    parser.add_argument("--download", action="store_true", help="Allow the official dataset download")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print("Loading official EMNIST ByClass training split...", flush=True)
    training = EMNIST(args.data_dir, split=config["dataset_split"], train=True, download=args.download)
    print("Loading official test split for class counts only...", flush=True)
    testing = EMNIST(args.data_dir, split=config["dataset_split"], train=False, download=args.download)
    if training.classes != testing.classes:
        raise ValueError("Training and test class mappings disagree")
    mapping = class_mapping(training.classes, config["classes"])
    train_targets, test_targets = training.targets.numpy(), testing.targets.numpy()
    train_indices, validation_indices = stratified_split(
        train_targets, mapping, config["validation_fraction"], config["seed"]
    )
    # Separate index namespace: these indices refer to the official TEST file.
    test_indices = np.flatnonzero(np.isin(test_targets, list(mapping)))
    if np.intersect1d(train_indices, validation_indices).size:
        raise RuntimeError("Training and validation partitions overlap")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_path = args.output_dir / "split_indices.npz"
    np.savez_compressed(split_path, train=train_indices, validation=validation_indices, test=test_indices)
    classes = []
    for source_id, label in mapping.items():
        classes.append({
            "character": config["classes"][label], "label": label, "source_label": source_id,
            "train": int(np.count_nonzero(train_targets[train_indices] == source_id)),
            "validation": int(np.count_nonzero(train_targets[validation_indices] == source_id)),
            "test": int(np.count_nonzero(test_targets[test_indices] == source_id)),
        })
    print("Hashing the four ByClass source files...", flush=True)
    raw = Path(training.raw_folder)
    sources = {
        path.name: sha256_file(path)
        for path in sorted(raw.glob("emnist-byclass-*-*-idx*-ubyte"))
    }
    if len(sources) != 4:
        raise RuntimeError("Expected four raw ByClass source files")
    report = {
        "dataset": "EMNIST ByClass", "source": "https://www.nist.gov/itl/products-and-services/emnist-dataset",
        "config": config, "classes": classes,
        "counts": {"train": len(train_indices), "validation": len(validation_indices), "test": len(test_indices)},
        "source_sha256": sources, "split_sha256": sha256_file(split_path),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy", "pillow")},
        "python_version": sys.version.split()[0],
        "test_policy": "Only counts are inspected; no model selection or preview uses test images.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "class_mapping.json").write_text(
        json.dumps({row["character"]: row["label"] for row in classes}, indent=2) + "\n", encoding="utf-8"
    )
    make_preview(training, train_indices, mapping, config["classes"], config["preprocessing"], args.output_dir / "samples.png")
    print("Class  train  validation  test")
    for row in classes:
        print(f"{row['character']:>5}  {row['train']:>5}  {row['validation']:>10}  {row['test']:>5}")
    print(f"Saved manifest, split indices, class mapping, and preview to {args.output_dir}")


if __name__ == "__main__":
    main()
