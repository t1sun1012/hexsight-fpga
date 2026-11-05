"""Integer-only CNN reference and an independent scalar model for future RTL.

Real value = stored integer / 2**fraction_bits. No floating-point operations
occur in either inference implementation. Quantization is an offline operation.
"""

import numpy as np

INPUT_BITS, INPUT_FRAC = 8, 8
WEIGHT_BITS, WEIGHT_FRAC = 16, 12
ACT_BITS, ACT_FRAC = 16, 8
BIAS_BITS, ACC_BITS = 32, 40
PRODUCT_FRAC = INPUT_FRAC + WEIGHT_FRAC
REQUANT_SHIFT = PRODUCT_FRAC - ACT_FRAC
ACT_MAX = (1 << (ACT_BITS - 1)) - 1
PARAMETERS = {
    "conv_weight": ((8, 1, 3, 3), WEIGHT_BITS, WEIGHT_FRAC),
    "conv_bias": ((8,), BIAS_BITS, PRODUCT_FRAC),
    "fc_weight": ((16, 392), WEIGHT_BITS, WEIGHT_FRAC),
    "fc_bias": ((16,), BIAS_BITS, PRODUCT_FRAC),
}


def format_spec():
    return {
        "version": 1,
        "input": {"bits": INPUT_BITS, "fraction_bits": INPUT_FRAC, "signed": False},
        "weight": {"bits": WEIGHT_BITS, "fraction_bits": WEIGHT_FRAC, "signed": True},
        "bias": {"bits": BIAS_BITS, "fraction_bits": PRODUCT_FRAC, "signed": True},
        "accumulator_and_logits": {"bits": ACC_BITS, "fraction_bits": PRODUCT_FRAC, "signed": True},
        "requantization_right_shift": REQUANT_SHIFT,
        "rounding": "nearest; exact half ties away from zero",
        "activation": {"bits": ACT_BITS, "fraction_bits": ACT_FRAC, "signed": True,
                       "relu_min": 0, "saturation_max": ACT_MAX},
        "parameter_overflow": "reject export; never clip",
        "accumulator_overflow": "reject; proven impossible for accepted parameters and uint8 inputs",
        "argmax_tie": "lowest class index",
        "convolution": "cross-correlation; stride 1; no padding; kernels not flipped",
        "pooling": "2x2 maximum; stride 2; no padding",
        "feature_order": "channel, row, column (column varies fastest)",
        "classes": "0123456789ABCDEF",
        "input_shape": [1, 16, 16],
        "conv_shape": [8, 14, 14],
        "pooled_shape": [8, 7, 7],
    }


def checked_signed(values, bits, name):
    values = np.asarray(values)
    if values.dtype.kind not in "iu" or values.size == 0:
        raise ValueError(f"{name} must contain integers")
    lower, upper = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if int(values.min()) < lower or int(values.max()) > upper:
        raise OverflowError(f"{name} does not fit signed {bits} bits")
    return values.astype(np.int64, copy=False)


def quantize(values, bits, fraction_bits):
    """Round binary64-scaled float parameters; fail instead of silently clipping."""
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Cannot quantize non-finite parameters")
    scaled = values * (1 << fraction_bits)
    rounded = np.copysign(np.floor(np.abs(scaled) + 0.5), scaled)
    if np.any(rounded < -(1 << (bits - 1))) or np.any(rounded > (1 << (bits - 1)) - 1):
        raise OverflowError(f"Quantized parameter does not fit signed {bits} bits")
    return rounded.astype(np.int64)


def round_shift(values, shift):
    """Signed nearest rounding, ties away from zero, for bounded int64 operands."""
    values = np.asarray(values)
    if values.dtype.kind not in "iu" or not 0 <= shift <= 40:
        raise ValueError("Expected integer operands and a shift from 0 to 40")
    values = checked_signed(values, 62, "rounding operand")
    if shift == 0:
        return values.copy()
    magnitude = (np.abs(values) + (1 << (shift - 1))) >> shift
    return np.where(values < 0, -magnitude, magnitude)


def relu_requantize(accumulators):
    checked_signed(accumulators, ACC_BITS, "convolution accumulator")
    return np.clip(round_shift(accumulators, REQUANT_SHIFT), 0, ACT_MAX)


def validate_pixels(pixels):
    pixels = np.asarray(pixels)
    if pixels.dtype != np.uint8 or pixels.ndim != 2 or pixels.shape[1] != 256 or not len(pixels):
        raise ValueError("Expected a nonempty uint8 batch shaped (N, 256)")
    return pixels


class FixedPointCNN:
    def __init__(self, parameters):
        if set(parameters) != set(PARAMETERS):
            raise ValueError("Expected convolution and dense weights/biases")
        self.parameters = {}
        for name, (shape, bits, _) in PARAMETERS.items():
            values = checked_signed(parameters[name], bits, name).copy()
            if values.shape != shape:
                raise ValueError(f"{name}: expected shape {shape}, got {values.shape}")
            values.setflags(write=False)
            self.parameters[name] = values
        self.bounds = self.prove_bounds()

    @classmethod
    def from_state_dict(cls, state_dict):
        names = {name: name.replace("_", ".") for name in PARAMETERS}
        if set(state_dict) != set(names.values()):
            raise ValueError("Checkpoint must contain only the supported CNN parameters")
        return cls({
            name: quantize(state_dict[key].detach().cpu().numpy(), bits, frac)
            for name, key in names.items()
            for _, bits, frac in [PARAMETERS[name]]
        })

    def prove_bounds(self):
        """Interval bounds cover ALL legal inputs and every partial sum, not samples."""
        p = self.parameters
        cw = p["conv_weight"].reshape(8, 9)
        conv_lower = p["conv_bias"] + 255 * np.minimum(cw, 0).sum(axis=1)
        conv_upper = p["conv_bias"] + 255 * np.maximum(cw, 0).sum(axis=1)
        # Requantization, ReLU, and max pooling are monotone.
        activation_upper = np.clip(round_shift(conv_upper, REQUANT_SHIFT), 0, ACT_MAX)
        feature_upper = np.repeat(activation_upper, 49)
        fc_lower = p["fc_bias"] + (np.minimum(p["fc_weight"], 0) * feature_upper).sum(axis=1)
        fc_upper = p["fc_bias"] + (np.maximum(p["fc_weight"], 0) * feature_upper).sum(axis=1)
        for values in (conv_lower, conv_upper, fc_lower, fc_upper):
            checked_signed(values, ACC_BITS, "all-input accumulator bounds")
        # Stronger format-only proof, including every signed 32-bit bias.
        conv_abs_bound = (1 << 31) + 9 * 255 * (1 << 15)
        fc_abs_bound = (1 << 31) + 392 * ACT_MAX * (1 << 15)
        if max(conv_abs_bound, fc_abs_bound) > (1 << (ACC_BITS - 1)) - 1:
            raise OverflowError("Accumulator format cannot contain all legal partial sums")
        return {
            "proof": "Bias plus all negative/all positive terms bounds every accumulation prefix.",
            "conv_lower_per_channel": conv_lower.tolist(),
            "conv_upper_per_channel": conv_upper.tolist(),
            "activation_upper_per_channel": activation_upper.tolist(),
            "fc_lower_per_class": fc_lower.tolist(),
            "fc_upper_per_class": fc_upper.tolist(),
            "format_only_conv_absolute_bound": conv_abs_bound,
            "format_only_fc_absolute_bound": fc_abs_bound,
            "signed_accumulator_limit": (1 << (ACC_BITS - 1)) - 1,
            "all_partial_sums_safe": True,
            "activation_saturation_possible_for_exported_weights":
                bool(np.any(round_shift(conv_upper, REQUANT_SHIFT) > ACT_MAX)),
        }

    def forward(self, pixels):
        pixels = validate_pixels(pixels)
        p = self.parameters
        images = pixels.astype(np.int64).reshape(-1, 16, 16)
        patches = np.lib.stride_tricks.sliding_window_view(images, (3, 3), axis=(1, 2))
        patches = patches.reshape(len(images), 196, 9)
        conv = patches @ p["conv_weight"].reshape(8, 9).T + p["conv_bias"]
        conv = conv.transpose(0, 2, 1).reshape(-1, 8, 14, 14)
        checked_signed(conv, ACC_BITS, "convolution accumulator")
        relu = relu_requantize(conv)
        pooled = relu.reshape(-1, 8, 7, 2, 7, 2).max(axis=(3, 5))
        logits = pooled.reshape(-1, 392) @ p["fc_weight"].T + p["fc_bias"]
        checked_signed(logits, ACC_BITS, "dense accumulator")
        return {"conv_acc": conv, "relu": relu, "pool": pooled, "logits": logits,
                "predictions": np.argmax(logits, axis=1)}


def scalar_forward(model, pixels):
    """Separate slow Python-int address/MAC loops; checks EVERY partial sum."""
    pixels = validate_pixels(pixels)
    p = model.parameters
    traces = {key: [] for key in ("conv_acc", "relu", "pool", "logits", "predictions")}
    low, high = -(1 << 39), (1 << 39) - 1
    for flat in pixels:
        conv = np.zeros((8, 14, 14), dtype=np.int64)
        relu = np.zeros_like(conv)
        for channel in range(8):
            for row in range(14):
                for col in range(14):
                    acc = int(p["conv_bias"][channel])
                    for kr in range(3):
                        for kc in range(3):
                            acc += int(flat[(row + kr) * 16 + col + kc]) * int(p["conv_weight"][channel, 0, kr, kc])
                            if not low <= acc <= high:
                                raise OverflowError("Scalar convolution overflow")
                    conv[channel, row, col] = acc
                    # ReLU makes signed negative rounding irrelevant to this output.
                    relu[channel, row, col] = min((acc + 2048) // 4096, 32767) if acc > 0 else 0
        features = []
        for channel in range(8):
            for row in range(7):
                for col in range(7):
                    features.append(max(int(relu[channel, 2 * row + dr, 2 * col + dc])
                                        for dr in range(2) for dc in range(2)))
        scores = []
        for output in range(16):
            acc = int(p["fc_bias"][output])
            for address, feature in enumerate(features):
                acc += feature * int(p["fc_weight"][output, address])
                if not low <= acc <= high:
                    raise OverflowError("Scalar dense overflow")
            scores.append(acc)
        winner = 0
        for output in range(1, 16):
            if scores[output] > scores[winner]:  # Strict > preserves the lowest tie index.
                winner = output
        for key, value in (("conv_acc", conv), ("relu", relu), ("pool", np.array(features).reshape(8, 7, 7)),
                           ("logits", scores), ("predictions", winner)):
            traces[key].append(value)
    return {key: np.asarray(value, dtype=np.int64) for key, value in traces.items()}
