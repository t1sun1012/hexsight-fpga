"""Verified, cached training/validation data. Never loads official test images."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from torchvision.datasets import EMNIST

from .data import class_mapping, preprocess_image


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_training_data(data_dir, audit_dir, config):
    data_dir, audit_dir = Path(data_dir), Path(audit_dir)
    audit = json.loads((audit_dir / "manifest.json").read_text(encoding="utf-8"))
    for field in ("classes", "dataset_split", "seed", "validation_fraction", "preprocessing"):
        if config[field] != audit["config"][field]:
            raise ValueError(f"Configuration field {field} differs from the audited dataset")
    if importlib.metadata.version("pillow") != audit["versions"]["pillow"]:
        raise ValueError("Use the audited Pillow version to keep preprocessing reproducible")
    split_path = audit_dir / "split_indices.npz"
    if sha256_file(split_path) != audit["split_sha256"]:
        raise ValueError("Saved split indices do not match the audit hash")
    raw_dir = data_dir / "EMNIST" / "raw"
    train_sources = {name: digest for name, digest in audit["source_sha256"].items() if "-train-" in name}
    for name, expected in train_sources.items():
        if sha256_file(raw_dir / name) != expected:
            raise ValueError(f"Source file changed: {name}")
    with np.load(split_path, allow_pickle=False) as split:
        train_indices, val_indices = split["train"], split["validation"]
    if len(np.unique(np.r_[train_indices, val_indices])) != len(train_indices) + len(val_indices):
        raise ValueError("Training/validation indices overlap or contain duplicates")
    cache_spec = {
        "sources": train_sources, "split_sha256": audit["split_sha256"],
        "classes": config["classes"], "preprocessing": config["preprocessing"],
        "pillow": audit["versions"]["pillow"],
        "preprocessing_code_sha256": sha256_file(Path(__file__).with_name("data.py")),
    }
    cache_key = hashlib.sha256(json.dumps(cache_spec, sort_keys=True).encode()).hexdigest()
    cache_dir = data_dir / "hexsight_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.npz"
    if cache_path.exists():
        print("Loading verified-source preprocessing cache...", flush=True)
        with np.load(cache_path, allow_pickle=False) as cached:
            pixels, labels = cached["pixels"], cached["labels"]
    else:
        source = EMNIST(data_dir, split=config["dataset_split"], train=True, download=False)
        mapping = class_mapping(source.classes, config["classes"])
        indices = np.r_[train_indices, val_indices]
        expected = np.flatnonzero(np.isin(source.targets.numpy(), list(mapping)))
        if not np.array_equal(np.sort(indices), expected):
            raise ValueError("Partitions do not cover the complete selected training subset")
        pixels = np.empty((len(indices), config["preprocessing"]["image_size"] ** 2), dtype=np.uint8)
        labels = np.empty(len(indices), dtype=np.int64)
        raw_images, raw_labels = source.data.numpy(), source.targets.numpy()
        for i, original_index in enumerate(indices):
            pixels[i] = preprocess_image(
                raw_images[original_index], config["preprocessing"],
                transpose=config["preprocessing"]["emnist_transpose"],
            ).reshape(-1)
            labels[i] = mapping[int(raw_labels[original_index])]
            if (i + 1) % 50000 == 0:
                print(f"Preprocessed {i + 1:,}/{len(indices):,} images", flush=True)
        # Cache writes are atomic so an interrupted preprocessing run cannot be reused.
        temporary = cache_path.with_suffix(".tmp.npz")
        np.savez(temporary, pixels=pixels, labels=labels)
        temporary.replace(cache_path)
    size = len(train_indices)
    expected_shape = (size + len(val_indices), config["preprocessing"]["image_size"] ** 2)
    if pixels.shape != expected_shape or pixels.dtype != np.uint8 or labels.shape != (expected_shape[0],):
        raise ValueError("Preprocessing cache shape/dtype mismatch")
    if labels.dtype != np.int64 or labels.min() < 0 or labels.max() >= len(config["classes"]):
        raise ValueError("Preprocessing cache contains invalid labels")
    for partition, subset in (("train", labels[:size]), ("validation", labels[size:])):
        if np.bincount(subset, minlength=len(config["classes"])).tolist() != [row[partition] for row in audit["classes"]]:
            raise ValueError(f"Cached {partition} class counts do not match audit")
    return {
        "train_pixels": pixels[:size], "train_labels": labels[:size], "train_indices": train_indices,
        "val_pixels": pixels[size:], "val_labels": labels[size:], "val_indices": val_indices,
        "cache_key": cache_key, "cache_sha256": sha256_file(cache_path),
        "split_sha256": audit["split_sha256"], "source_sha256": train_sources,
    }
