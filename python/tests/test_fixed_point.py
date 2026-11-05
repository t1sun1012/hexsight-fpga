"""Arithmetic boundaries and independent data/address checks for the FPGA contract."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hexsight.fixed_point import (
    ACT_MAX, FixedPointCNN, PARAMETERS, quantize, relu_requantize, round_shift, scalar_forward,
)
from hexsight.fixed_point_io import export_bundle, load_bundle, read_mem, write_mem


def zero_parameters():
    return {name: np.zeros(shape, dtype=np.int64) for name, (shape, _, _) in PARAMETERS.items()}


class FixedPointTests(unittest.TestCase):
    def test_parameter_rounding_ties_and_range(self):
        np.testing.assert_array_equal(quantize(np.array([-1.5, -0.5, 0, 0.5, 1.5]) / 4096, 16, 12),
                                      [-2, -1, 0, 1, 2])
        np.testing.assert_array_equal(quantize([-8, 32767 / 4096], 16, 12), [-32768, 32767])
        for values in ([8], [-8.001], [np.nan], [np.inf]):
            with self.assertRaises((ValueError, OverflowError)):
                quantize(values, 16, 12)

    def test_signed_round_shift_ties_and_neighbors(self):
        np.testing.assert_array_equal(
            round_shift([-6144, -6143, -2049, -2048, -2047, -1, 0, 2047, 2048, 2049, 6143, 6144], 12),
            [-2, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 2],
        )
        np.testing.assert_array_equal(round_shift([-5, 7], 0), [-5, 7])
        with self.assertRaises(ValueError):
            round_shift([0.5], 12)
        with self.assertRaises(ValueError):
            round_shift([1], -1)

    def test_relu_saturation_boundaries(self):
        boundary = ACT_MAX * 4096
        np.testing.assert_array_equal(
            relu_requantize(np.array([-(1 << 39), -2048, 0, 2047, 2048,
                                      boundary - 2048, boundary + 2047, boundary + 2048, (1 << 39) - 1])),
            [0, 0, 0, 0, 1, ACT_MAX, ACT_MAX, ACT_MAX, ACT_MAX],
        )
        with self.assertRaises(OverflowError):
            relu_requantize(np.array([1 << 39], dtype=np.int64))

    def test_hand_computed_cross_correlation_bias_pool_and_feature_order(self):
        p = zero_parameters()
        p["conv_weight"][0, 0, 0, 0] = 4096
        p["conv_bias"][0] = 2048  # Half an activation LSB: always rounds upward here.
        p["conv_weight"][1, 0, 2, 2] = 8192
        p["conv_bias"][1] = -2048
        p["fc_weight"][0, 0] = 4096
        p["fc_weight"][1, 49] = -4096
        p["fc_weight"][2, 48] = 4096
        p["fc_bias"][2] = -1
        model = FixedPointCNN(p)
        trace = model.forward(np.arange(256, dtype=np.uint8).reshape(1, 256))
        self.assertEqual(trace["conv_acc"][0, 0, 0, 0], 2048)
        self.assertEqual(trace["conv_acc"][0, 1, 0, 0], 34 * 8192 - 2048)
        np.testing.assert_array_equal(trace["pool"][0, 0, 0, :3], [18, 20, 22])
        self.assertEqual(trace["pool"][0, 0, -1, -1], 222)
        self.assertEqual(trace["pool"][0, 1, 0, 0], 102)
        np.testing.assert_array_equal(trace["logits"][0, :3], [18 * 4096, -102 * 4096, 222 * 4096 - 1])
        self.assertEqual(trace["predictions"][0], 2)

    def test_negative_scores_and_lowest_index_tie(self):
        p = zero_parameters()
        p["fc_bias"][:] = -50
        p["fc_bias"][[3, 7]] = -2
        model = FixedPointCNN(p)
        pixels = np.zeros((1, 256), dtype=np.uint8)
        self.assertEqual(model.forward(pixels)["predictions"][0], 3)
        self.assertEqual(scalar_forward(model, pixels)["predictions"][0], 3)

    def test_random_extremes_match_independent_scalar_at_every_stage(self):
        rng = np.random.default_rng(311)
        p = {name: rng.integers(-(1 << (bits - 1)), 1 << (bits - 1), shape, dtype=np.int64)
             for name, (shape, bits, _) in PARAMETERS.items()}
        model = FixedPointCNN(p)
        pixels = rng.integers(0, 256, (2, 256), dtype=np.uint8)
        fast, slow = model.forward(pixels), scalar_forward(model, pixels)
        for stage in fast:
            np.testing.assert_array_equal(fast[stage], slow[stage], err_msg=stage)

    def test_format_extremes_prove_prefix_safety_and_saturate(self):
        p = zero_parameters()
        p["conv_weight"].fill(32767)
        p["conv_bias"].fill((1 << 31) - 1)
        p["fc_weight"].fill(-32768)
        p["fc_bias"].fill(-(1 << 31))
        model = FixedPointCNN(p)
        self.assertTrue(model.bounds["all_partial_sums_safe"])
        self.assertTrue(model.bounds["activation_saturation_possible_for_exported_weights"])
        self.assertLess(model.bounds["format_only_fc_absolute_bound"], 1 << 39)
        trace = model.forward(np.full((1, 256), 255, dtype=np.uint8))
        self.assertTrue(np.all(trace["pool"] == 32767))
        expected = -(1 << 31) - 392 * 32767 * 32768
        self.assertTrue(np.all(trace["logits"] == expected))
        for stage, values in scalar_forward(model, np.full((1, 256), 255, dtype=np.uint8)).items():
            np.testing.assert_array_equal(values, trace[stage])

    def test_invalid_pixels_parameters_and_shapes_rejected(self):
        model = FixedPointCNN(zero_parameters())
        for pixels in (np.zeros((1, 256)), np.zeros((0, 256), dtype=np.uint8),
                       np.zeros((16, 16), dtype=np.uint8), np.zeros((1, 255), dtype=np.uint8)):
            with self.assertRaises(ValueError):
                model.forward(pixels)
        p = zero_parameters()
        p["conv_weight"][0, 0, 0, 0] = 32768
        with self.assertRaises(OverflowError):
            FixedPointCNN(p)
        p = zero_parameters()
        p["fc_weight"] = np.zeros((16, 391), dtype=np.int64)
        with self.assertRaises(ValueError):
            FixedPointCNN(p)

    def test_twos_complement_memory_encoding_and_strict_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "words.mem"
            values = np.array([-32768, -1, 0, 32767], dtype=np.int64)
            write_mem(path, values, 16)
            self.assertEqual(path.read_text(), "8000\nFFFF\n0000\n7FFF\n")
            np.testing.assert_array_equal(read_mem(path, 16, (4,)), values)
            values40 = np.array([-(1 << 39), -1, (1 << 39) - 1])
            write_mem(path, values40, 40)
            np.testing.assert_array_equal(read_mem(path, 40, (3,)), values40)
            for text in ("FFF\n", "FFFF\n0000\n", "GGGG\n", " FFFF\n"):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    read_mem(path, 16, (1,))
            with self.assertRaises(OverflowError):
                write_mem(path, np.array([256]), 8, signed=False)

    def test_model_export_round_trip_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            p = zero_parameters()
            p["conv_weight"][0, 0, 0, 0] = -32768
            model = FixedPointCNN(p)
            manifest = export_bundle(directory, model, {"test": True})
            restored, _ = load_bundle(directory)
            self.assertEqual(manifest["storage_bytes"], 12784)
            for name in p:
                np.testing.assert_array_equal(restored.parameters[name], p[name])
            path = Path(directory) / "conv_weight.mem"
            path.write_text(path.read_text().replace("8000", "8001", 1))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_bundle(directory)

    def test_manifest_layout_cannot_override_hardware_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            export_bundle(directory, FixedPointCNN(zero_parameters()), {})
            path = Path(directory) / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["files"]["conv_weight"]["file"] = "../unexpected.mem"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Invalid memory layout"):
                load_bundle(directory)


if __name__ == "__main__":
    unittest.main()
