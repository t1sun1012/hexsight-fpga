"""Guard against label, orientation, and split mistakes before training."""

import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hexsight.data import class_mapping, preprocess_image, stratified_split

SETTINGS = json.loads((Path(__file__).resolve().parents[2] / "configs" / "baseline.json").read_text())["preprocessing"]


class DatasetTests(unittest.TestCase):
    def test_mapping_uses_names_not_label_order(self):
        self.assertEqual(class_mapping(["a", "F", "0", "A"], "0AF"), {2: 0, 3: 1, 1: 2})

    def test_split_is_disjoint_complete_stratified_and_reproducible(self):
        targets = np.repeat([5, 9, 12], 20)
        mapping = {5: 0, 9: 1}
        train, validation = stratified_split(targets, mapping, 0.15, 311)
        self.assertEqual(len(train), 34)
        self.assertEqual(len(validation), 6)
        self.assertEqual(np.intersect1d(train, validation).size, 0)
        np.testing.assert_array_equal(np.sort(np.r_[train, validation]), np.arange(40))
        self.assertEqual(np.count_nonzero(targets[validation] == 5), 3)
        repeated = stratified_split(targets, mapping, 0.15, 311)
        np.testing.assert_array_equal(train, repeated[0])
        np.testing.assert_array_equal(validation, repeated[1])

    def test_emnist_orientation_matches_upright_canvas(self):
        upright = np.zeros((28, 28), dtype=np.uint8)
        upright[4:24, 6:9] = 255
        upright[4:7, 6:22] = 255
        upright[12:15, 6:17] = 255  # Asymmetric F makes transpose errors visible.
        emnist = preprocess_image(upright.T, SETTINGS, transpose=True)
        canvas = preprocess_image(upright, SETTINGS)
        np.testing.assert_array_equal(emnist, canvas)
        self.assertFalse(np.array_equal(canvas, preprocess_image(upright.T, SETTINGS)))
        self.assertEqual(canvas.shape, (16, 16))
        self.assertEqual(canvas.dtype, np.uint8)
        self.assertEqual(int(canvas[:2].sum() + canvas[-2:].sum()), 0)

    def test_blank_and_normalization(self):
        blank = preprocess_image(np.zeros((28, 28), dtype=np.uint8), SETTINGS)
        self.assertEqual(int(blank.sum()), 0)
        pixel = np.array([0, 255], dtype=np.uint8).astype(np.float32) / SETTINGS["normalization_divisor"]
        np.testing.assert_array_equal(pixel, [0, 255 / 256])

    def test_non_byte_input_rejected(self):
        with self.assertRaises(ValueError):
            preprocess_image(np.ones((28, 28), dtype=np.float32), SETTINGS)


if __name__ == "__main__":
    unittest.main()
