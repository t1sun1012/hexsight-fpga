"""Check spatial arithmetic, flattening order, training, and legacy compatibility."""

import io
import json
from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hexsight.model import HexSightCNN, HexSightMLP, architecture_details, build_model

REPO = Path(__file__).resolve().parents[2]


class CNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_hand_computed_convolution_pooling_and_channel_order(self):
        # Asymmetric 2x2 kernel checks that cross-correlation does not flip it.
        model = HexSightCNN(image_size=3, filters=2, kernel_size=2, pool_size=2, num_classes=2)
        with torch.no_grad():
            model.conv.weight.copy_(torch.tensor([[[[1., 2.], [3., 4.]]], [[[-1., -2.], [-3., -4.]]]]))
            model.conv.bias.copy_(torch.tensor([1., 0.]))
            model.fc.weight.copy_(torch.eye(2))
            model.fc.bias.copy_(torch.tensor([0.5, -0.5]))
        pixels = torch.arange(1., 10.).reshape(1, 9)
        # First channel: [37,47;67,77] + bias 1 -> max 78.
        # Second channel is negative throughout -> ReLU -> zero.
        torch.testing.assert_close(model(pixels), torch.tensor([[78.5, -0.5]]), rtol=0, atol=0)

    def test_spatial_flatten_order(self):
        model = HexSightCNN(image_size=5, filters=1, kernel_size=2, pool_size=2, num_classes=4)
        with torch.no_grad():
            model.conv.weight.zero_()
            model.conv.weight[0, 0, 0, 0] = 1
            model.conv.bias.zero_()
            model.fc.weight.copy_(torch.eye(4))
            model.fc.bias.zero_()
        # Top-left pixel of each 2x2 patch, then 2x2 max pooling.
        torch.testing.assert_close(
            model(torch.arange(25.).reshape(1, 25)),
            torch.tensor([[6., 8., 16., 18.]]), rtol=0, atol=0,
        )

    def test_baseline_dimensions_and_cost(self):
        config = json.loads((REPO / "configs/cnn.json").read_text())
        model = build_model(config)
        self.assertEqual(tuple(model(torch.zeros(2, 256)).shape), (2, 16))
        details = architecture_details(model)
        self.assertEqual(details["parameters"], 6368)
        self.assertEqual(details["weights"], 6344)
        self.assertEqual(details["macs_per_image"], 20384)
        self.assertEqual(details["pool_comparisons_per_image"], 1176)
        self.assertEqual(details["pooled_shape_chw"], [8, 7, 7])

    def test_legacy_mlp_config_still_loads(self):
        config = json.loads((REPO / "configs/baseline.json").read_text())
        model = build_model(config)
        self.assertIsInstance(model, HexSightMLP)
        self.assertEqual(architecture_details(model)["parameters"], 8752)
        self.assertEqual(architecture_details(model)["macs_per_image"], 8704)

    def test_cnn_learns_spatial_patterns_and_reloads(self):
        torch.manual_seed(311)
        model = HexSightCNN(image_size=6, filters=4, kernel_size=3, pool_size=2, num_classes=2)
        images = torch.zeros(2, 6, 6)
        images[0, 1:5, 1:3] = 1  # Vertical stroke
        images[1, 1:3, 1:5] = 1  # Horizontal stroke
        inputs, labels = images.flatten(1), torch.tensor([0, 1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        for _ in range(100):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(inputs), labels)
            loss.backward()
            optimizer.step()
        torch.testing.assert_close(model(inputs).argmax(1), labels)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        loaded = HexSightCNN(image_size=6, filters=4, kernel_size=3, pool_size=2, num_classes=2)
        loaded.load_state_dict(torch.load(buffer, weights_only=True))
        torch.testing.assert_close(loaded(inputs), model(inputs), rtol=0, atol=0)

    def test_rejects_invalid_image_and_model_shapes(self):
        with self.assertRaises(ValueError):
            HexSightCNN()(torch.zeros(1, 512))
        with self.assertRaises(ValueError):
            HexSightCNN(image_size=4, kernel_size=2, pool_size=2)


if __name__ == "__main__":
    unittest.main()
