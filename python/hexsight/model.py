"""Small fully connected classifier intended for later fixed-point export."""

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


def build_model(config):
    if config["model"]["activation"] != "relu":
        raise ValueError("This baseline implements ReLU only")
    return HexSightMLP(
        config["preprocessing"]["image_size"] ** 2,
        config["model"]["hidden_units"],
        len(config["classes"]),
    )
