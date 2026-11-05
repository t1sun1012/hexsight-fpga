"""Check metric semantics, loss reduction, and trainable network behavior."""

import io
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hexsight.metrics import classification_metrics, inverse_frequency_weights
from hexsight.model import HexSightMLP
from train import evaluate


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_metrics_count_rows_as_true_classes(self):
        metrics = classification_metrics([0, 0, 0, 1], [0, 0, 1, 1], "AB")
        self.assertEqual(metrics["confusion_matrix"], [[2, 1], [0, 1]])
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["macro_recall"], (2 / 3 + 1) / 2)
        self.assertEqual(metrics["top_confusions_by_count"][0]["true"], "A")

    def test_class_weights_equalize_total_training_weight(self):
        labels = torch.tensor([0, 0, 0, 1])
        weights = inverse_frequency_weights(labels, 2)
        self.assertAlmostEqual(float(weights[labels[:3]].sum()), float(weights[labels[3:]].sum()))
        with self.assertRaises(ValueError):
            inverse_frequency_weights(torch.tensor([0, 0]), 2)

    def test_weighted_evaluation_uses_global_denominator(self):
        torch.manual_seed(2)
        model = HexSightMLP(4, 5, 2)
        inputs, labels = torch.rand(5, 4), torch.tensor([0, 0, 1, 0, 1])
        weights = torch.tensor([0.5, 2.0])
        result, predictions = evaluate(model, inputs, labels, weights, "AB", batch_size=2)
        expected = F.cross_entropy(model(inputs), labels, weight=weights).item()
        self.assertAlmostEqual(result["weighted_loss"], expected, places=6)
        np.testing.assert_array_equal(predictions, model(inputs).argmax(1).detach().numpy())

    def test_network_can_fit_a_tiny_labeled_problem_and_reload(self):
        torch.manual_seed(311)
        model = HexSightMLP(4, 8, 4)
        inputs = torch.eye(4)
        labels = torch.arange(4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        for _ in range(120):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(inputs), labels)
            loss.backward()
            optimizer.step()
        torch.testing.assert_close(model(inputs).argmax(1), labels)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        loaded = HexSightMLP(4, 8, 4)
        loaded.load_state_dict(torch.load(buffer, weights_only=True))
        torch.testing.assert_close(loaded(inputs), model(inputs), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
