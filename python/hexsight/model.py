"""Compact classifiers for handwritten hexadecimal recognition."""

import torch
from torch import nn


class HexSightMLP(nn.Module):
    def __init__(self, input_size=256, hidden_units=32, num_classes=16):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_units)
        self.fc2 = nn.Linear(hidden_units, num_classes)

    def forward(self, inputs):
        # Inputs are float32 pixels / 256, shaped (batch, 256).
        # Return raw logits; cross-entropy applies its own log-softmax.
        return self.fc2(torch.relu(self.fc1(inputs)))


class HexSightCNN(nn.Module):
    def __init__(self, image_size=16, filters=8, kernel_size=3, pool_size=2, num_classes=16):
        super().__init__()
        if min(image_size, filters, kernel_size, pool_size, num_classes) < 1:
            raise ValueError("All network dimensions must be positive")
        conv_side = image_size - kernel_size + 1
        if conv_side < pool_size or conv_side % pool_size:
            raise ValueError("Valid convolution must produce complete pooling windows")
        self.image_size = image_size
        self.conv_side = conv_side
        self.pooled_side = conv_side // pool_size
        self.conv = nn.Conv2d(1, filters, kernel_size, stride=1, padding=0)
        self.pool = nn.MaxPool2d(pool_size, stride=pool_size)
        self.fc = nn.Linear(filters * self.pooled_side ** 2, num_classes)

    def forward(self, inputs):
        if inputs.ndim != 2 or inputs.shape[1] != self.image_size ** 2:
            raise ValueError("Expected a batch of flattened grayscale images")
        images = inputs.reshape(inputs.shape[0], 1, self.image_size, self.image_size)
        features = self.pool(torch.relu(self.conv(images)))
        # NCHW layout: flatten channel first, then row, then column.
        # PyTorch Conv2d uses cross-correlation; kernels are not flipped.
        return self.fc(features.flatten(start_dim=1))


def build_model(config):
    settings = config["model"]
    if settings["activation"] != "relu":
        raise ValueError("These classifiers implement ReLU only")
    model_type = settings.get("type", "mlp")  # Original checkpoints predate this field.
    if model_type == "mlp":
        return HexSightMLP(
            config["preprocessing"]["image_size"] ** 2,
            settings["hidden_units"],
            len(config["classes"]),
        )
    if model_type == "cnn":
        return HexSightCNN(
            config["preprocessing"]["image_size"], settings["filters"],
            settings["kernel_size"], settings["pool_size"], len(config["classes"]),
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def architecture_details(model):
    details = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "weights": sum(p.numel() for name, p in model.named_parameters() if name.endswith("weight")),
        "mac_count_note": "Arithmetic estimate; excludes bias adds, ReLU, pooling, and memory/control overhead.",
    }
    if isinstance(model, HexSightCNN):
        conv_macs = model.conv.out_channels * model.conv_side ** 2 * model.conv.kernel_size[0] ** 2
        dense_macs = model.fc.in_features * model.fc.out_features
        details.update({
            "type": "cnn", "input_shape_chw": [1, model.image_size, model.image_size],
            "convolution_shape_chw": [model.conv.out_channels, model.conv_side, model.conv_side],
            "pooled_shape_chw": [model.conv.out_channels, model.pooled_side, model.pooled_side],
            "flattened_features": model.fc.in_features,
            "macs_per_image": conv_macs + dense_macs,
            "pool_comparisons_per_image": model.fc.in_features * (model.pool.kernel_size ** 2 - 1),
        })
    elif isinstance(model, HexSightMLP):
        details.update({
            "type": "mlp",
            "macs_per_image": model.fc1.in_features * model.fc1.out_features + model.fc2.in_features * model.fc2.out_features,
            "pool_comparisons_per_image": 0,
        })
    else:
        raise TypeError("Unknown classifier")
    return details
