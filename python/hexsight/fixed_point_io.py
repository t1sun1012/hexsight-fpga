"""Strict hexadecimal memory interchange for Verilog $readmemh."""

import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .fixed_point import FixedPointCNN, PARAMETERS, checked_signed, format_spec


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def write_mem(path, values, bits, signed=True):
    values = np.asarray(values)
    if signed:
        values = checked_signed(values, bits, "memory word")
    elif values.dtype.kind not in "iu" or values.size == 0 or int(values.min()) < 0 or int(values.max()) >= 1 << bits:
        raise OverflowError("Unsigned memory word out of range")
    digits, mask = (bits + 3) // 4, (1 << bits) - 1
    text = "".join(f"{int(value) & mask:0{digits}X}\n" for value in values.flat)
    Path(path).write_text(text, encoding="ascii", newline="\n")


def read_mem(path, bits, shape, signed=True):
    words = Path(path).read_text(encoding="ascii").splitlines()
    digits = (bits + 3) // 4
    if len(words) != int(np.prod(shape)) or any(re.fullmatch(rf"[0-9a-fA-F]{{{digits}}}", word) is None for word in words):
        raise ValueError(f"Invalid memory word width/count: {path}")
    raw = [int(word, 16) for word in words]
    if any(value >= 1 << bits for value in raw):
        raise ValueError("Nonzero padding bits in memory word")
    values = [value - (1 << bits) if signed and value >= 1 << (bits - 1) else value for value in raw]
    return np.array(values, dtype=np.int64).reshape(shape)



def rtl_constants():
    spec = format_spec()
    values = {
        "HS_INPUT_BITS": spec["input"]["bits"], "HS_INPUT_FRAC": spec["input"]["fraction_bits"],
        "HS_WEIGHT_BITS": spec["weight"]["bits"], "HS_WEIGHT_FRAC": spec["weight"]["fraction_bits"],
        "HS_BIAS_BITS": spec["bias"]["bits"], "HS_PRODUCT_FRAC": spec["bias"]["fraction_bits"],
        "HS_ACT_BITS": spec["activation"]["bits"], "HS_ACT_FRAC": spec["activation"]["fraction_bits"],
        "HS_ACC_BITS": spec["accumulator_and_logits"]["bits"],
        "HS_REQUANT_SHIFT": spec["requantization_right_shift"],
        "HS_IMAGE_SIDE": 16, "HS_KERNEL_SIDE": 3, "HS_CHANNELS": 8,
        "HS_CONV_SIDE": 14, "HS_POOL_SIDE": 7, "HS_FEATURES": 392, "HS_CLASSES": 16,
    }
    return ("// Generated numerical contract. Include inside a module, not at file scope.\n"
            + "".join(f"localparam integer {name} = {value};\n" for name, value in values.items()))


def export_bundle(directory, model, provenance):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, (shape, bits, frac) in PARAMETERS.items():
        filename = name + ".mem"
        write_mem(directory / filename, model.parameters[name], bits)
        files[name] = {
            "file": filename, "shape": list(shape), "bits": bits, "fraction_bits": frac,
            "signed": True, "words": int(np.prod(shape)), "sha256": sha256(directory / filename),
        }
    (directory / "fixed_point.vh").write_text(rtl_constants(), encoding="ascii", newline="\n")
    manifest = {
        "rtl_constants": {"file": "fixed_point.vh", "sha256": sha256(directory / "fixed_point.vh")},
        "format": format_spec(), "files": files, "provenance": provenance,
        "bounds": model.bounds,
        "storage_bytes": sum(int(np.prod(shape)) * bits // 8 for shape, bits, _ in PARAMETERS.values()),
        "address_order": {
            "conv_weight": "channel * 9 + kernel_row * 3 + kernel_col",
            "conv_bias": "channel",
            "fc_weight": "class * 392 + channel * 49 + pooled_row * 7 + pooled_col",
            "fc_bias": "class",
        },
    }
    write_json(directory / "manifest.json", manifest)
    return manifest


def load_bundle(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format"] != format_spec() or set(manifest["files"]) != set(PARAMETERS):
        raise ValueError("Unsupported fixed-point memory format")
    header = manifest["rtl_constants"]
    if (header["file"] != "fixed_point.vh"
            or header["sha256"] != sha256(directory / "fixed_point.vh")
            or (directory / "fixed_point.vh").read_bytes() != rtl_constants().encode("ascii")):
        raise ValueError("RTL constants differ from the fixed-point contract")
    parameters = {}
    for name, (shape, bits, frac) in PARAMETERS.items():
        spec = manifest["files"][name]
        filename = name + ".mem"
        expected = {"file": filename, "shape": list(shape), "bits": bits, "fraction_bits": frac,
                    "signed": True, "words": int(np.prod(shape))}
        if any(spec.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Invalid memory layout for {name}")
        if sha256(directory / filename) != spec["sha256"]:
            raise ValueError(f"Memory hash mismatch for {name}")
        parameters[name] = read_mem(directory / filename, bits, shape)
    model = FixedPointCNN(parameters)
    if model.bounds != manifest["bounds"]:
        raise ValueError("Saved and recomputed accumulator proofs differ")
    return model, manifest
