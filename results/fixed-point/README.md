# Fixed-point CNN validation report

Completed November 5, 2026. Post-training conversion of the Step 2 CNN, without retraining or test-set access.

| Measure | Float32 CNN | Fixed-point CNN |
|---|---:|---:|
| Validation images | 57,612 | 57,612 |
| Correct predictions | 56,151 | 56,154 |
| Accuracy | 97.464070% | 97.469277% |
| Macro recall | 97.010720% | 97.014347% |

The integer model uses signed 16-bit weights with 12 fractional bits, signed 16-bit activations with 8 fractional bits, signed 32-bit biases, and signed 40-bit accumulators. Biases and accumulators have 20 fractional bits. Inputs remain unsigned bytes representing pixel / 256.

Prediction agreement with the floating-point checkpoint is **99.986114%**: 8 differences, consisting of 5 fixed errors, 2 introduced errors, and 1 changed wrong answer. The +0.005207 percentage-point accuracy change indicates essentially preserved accuracy, not a demonstrated general improvement.

The evaluated model was reconstructed from the exported FPGA .mem files. It did not use floating-point arithmetic for convolution, activation rescaling, pooling, dense inference, or argmax.

Observed activation saturation: **0 of 90,335,616 activations**. The model-specific analytical proof also rules out activation saturation for every legal uint8 input. All accumulation prefixes fit the 40-bit format.

Mean absolute score error relative to float32 is 0.009368; maximum is 0.059006, measured in real score units. The scores are logits, not probabilities.

Files:

- metrics.json: classification metrics, float comparison, observed integer ranges, and score digest.
- per_class.csv: class support, recall, precision, and F1.
- validation_predictions.npz: frozen source indices, true labels, integer predictions, and floating-point predictions.
- run.json: checkpoint, data, software/source provenance, and exported-artifact hashes.
- verification.json: fresh-process comparison of all 57,612 predictions, 921,792 scores, source parameters, and 24 scalar-checked RTL vector cases.

All **26 tests pass**. The full verification has zero mismatches. Stored evaluation time covers both implementations and diagnostics, so it is not an inference-speed comparison.

See [the complete Step 3 guide](../../docs/step-03.md), [FPGA parameters](../../model/cnn-fixed-point/README.md), and [RTL vectors](../../test_vectors/cnn-fixed-point/README.md).

The official test set remains unused. The next milestone is Verilog implementation and bit-for-bit simulation against these vectors.
