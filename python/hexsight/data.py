"""Explicit class mapping, image preprocessing, and reproducible data splits."""

import numpy as np
from PIL import Image


def class_mapping(source_classes, desired_classes):
    """Look up class names rather than assuming source label IDs."""
    if len(set(desired_classes)) != len(desired_classes):
        raise ValueError("Desired class names must be unique")
    return {source_classes.index(name): i for i, name in enumerate(desired_classes)}


def preprocess_image(pixels, settings, *, transpose=False):
    """Return uint8 pixels; divide by 256 only when feeding the model.

    EMNIST raw storage needs transpose=True. An upright drawing uses False.
    Bounding-box detection uses >= threshold. Padding is clipped to the image.
    Resize dimensions round half up; odd spare padding goes right/bottom.
    A blank image remains blank. The host UI will reject blanks separately.
    """
    pixels = np.asarray(pixels)
    if pixels.ndim != 2 or pixels.dtype != np.uint8:
        raise ValueError("Expected a two-dimensional uint8 grayscale image")
    if not all(pixels.shape):
        raise ValueError("Image cannot have an empty dimension")
    if transpose:
        pixels = pixels.T
    size = settings["image_size"]
    foreground_size = settings["foreground_size"]
    if not 1 <= foreground_size <= size:
        raise ValueError("Foreground size must fit inside the output image")
    canvas = Image.new("L", (size, size), 0)
    ys, xs = np.nonzero(pixels >= settings["foreground_threshold"])
    if not len(xs):
        return np.asarray(canvas).copy()
    pad = settings["bounding_box_padding"]
    left, top = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)
    right = min(pixels.shape[1], int(xs.max()) + pad + 1)
    bottom = min(pixels.shape[0], int(ys.max()) + pad + 1)
    crop = Image.fromarray(pixels[top:bottom, left:right])
    scale = foreground_size / max(crop.size)
    width, height = (max(1, int(d * scale + 0.5)) for d in crop.size)
    resized = crop.resize((width, height), Image.Resampling.BILINEAR)
    canvas.paste(resized, ((size - width) // 2, (size - height) // 2))
    return np.asarray(canvas).copy()


def stratified_split(targets, mapping, fraction, seed):
    """Partition official TRAIN indices only, retaining every selected sample."""
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    targets = np.asarray(targets)
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for source_id in mapping:
        indices = np.flatnonzero(targets == source_id)
        if len(indices) < 2:
            raise ValueError(f"Class {source_id} needs at least two samples")
        rng.shuffle(indices)
        count = min(len(indices) - 1, max(1, int(len(indices) * fraction + 0.5)))
        validation.append(indices[:count])
        train.append(indices[count:])
    train, validation = np.concatenate(train), np.concatenate(validation)
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation
