# Step 3: convert the CNN to fixed-point arithmetic

Completed November 5, 2026. The model from [Step 2](step-02.md) now has an integer-only inference reference, exported FPGA memory files, and independently checked intermediate test vectors.

No retraining was needed. This is post-training quantization: represent each learned floating-point parameter as an integer with a fixed scale. The official test set remains unused. [Step 4](step-04.md) now records RTL simulation and standalone implementation checks; board integration and measurements remain upcoming.

## What fixed-point means

A stored integer represents a real value divided by a power of two:

    real_value = stored_integer / 2^fraction_bits

For example, the weight 0.5 is stored as 2048 because weights have 12 fractional bits: 2048 / 4096 = 0.5.
The input byte 128 still means 128 / 256 = 0.5. Their integer product is 262,144, which represents 0.25 at scale 2^20.

The FPGA can implement these calculations using integer multiplication, addition, comparison, and shifts. It does not need floating-point arithmetic or division hardware for this inference path.

Training and image resizing stay on the laptop. The FPGA receives the already-preprocessed 256 pixel bytes.

## Frozen numerical contract

The CNN topology remains:

    uint8 pixels [16 x 16]
      -> valid 3x3 cross-correlation, 8 channels
      -> rounded rescaling + ReLU + saturation [8 x 14 x 14]
      -> 2x2 max pooling [8 x 7 x 7]
      -> channel-major flatten [392]
      -> dense layer [16 signed scores]
      -> argmax [class index 0..15]

| Quantity | Storage width | Fractional bits | Real value |
|---|---:|---:|---|
| Input pixel | unsigned 8 | 8 | p / 256 |
| Convolution and dense weights | signed 16 | 12 | w / 4096 |
| Convolution and dense biases | signed 32 | 20 | b / 1,048,576 |
| ReLU and pooled activations | signed 16, restricted to nonnegative | 8 | q / 256 |
| MAC accumulator and output score | signed 40 | 20 | a / 1,048,576 |

Both input and activation operands are extended to signed 16 bits before multiplication. Pixels must be **zero-extended**, so values 128-255 remain positive. A signed 16x16 product fits signed 32 bits and must be sign-extended to 40 bits before accumulation.

Weights cover [-8, 7.999755859375]. The trained weights range approximately from -4.7323 to 3.5130 in the dense layer and -4.6287 to 2.9767 in convolution, so this format fits without clipping. Activations cover [0, 127.99609375].

These are explicit width/fraction definitions; they avoid ambiguous Q-format naming conventions. The CNN contract supersedes the scales proposed in the earlier MLP plan.

## Exact arithmetic

Offline export:

    W = round_nearest_ties_away_from_zero(float_weight * 4096)
    B = round_nearest_ties_away_from_zero(float_bias * 1048576)

Export fails if a parameter is non-finite or outside its integer range. There is no silent parameter clipping.

For channel c and convolution output position (r, s):

    a = B_conv[c]
    for kr = 0..2:
        for kc = 0..2:
            a += pixel[r + kr, s + kc] * W_conv[c, kr, kc]

    q = clamp(round_nearest_ties_away_from_zero(a / 4096), 0, 32767)

The convolution is cross-correlation: do not flip the kernel. There is no padding and stride is one. Bias is added once per output, at product scale. Accumulation does not truncate or round after each multiplication.

Rounding examples for a right shift of 12:

| Accumulator integer | Rounded activation integer, before ReLU |
|---:|---:|
| 2047 | 0 |
| 2048 | 1 |
| 6144 | 2 |
| -2047 | 0 |
| -2048 | -1 |
| -6144 | -2 |

A signed arithmetic shift alone is not this rounding rule. For the convolution activation, RTL can handle a <= 0 as zero, then calculate (a + 2048) >> 12 for positive a and saturate to 32767. Use an extended temporary for the rounding addition, such as 41 bits, before clipping and narrowing.

Max pooling selects the largest of each non-overlapping 2x2 group of q values. It changes neither scale nor width.

Flattening order is channel, row, column, with column varying fastest:

    feature_index = channel * 49 + pooled_row * 7 + pooled_column

Dense inference:

    score[class] = B_fc[class]
    for feature = 0..391:
        score[class] += pooled[feature] * W_fc[class, feature]

Scores remain signed 40-bit integers at scale 2^20. Do not apply ReLU, rescale, or add softmax to the output layer. Argmax compares signed scores. Initialize the winner with class 0's score, even if it is negative; replace it only for a strictly greater score. Exact ties choose the lowest class index.

Classes are ordered 0123456789ABCDEF. Seven-segment rendering of B and D as b and d happens later in the display logic.

## Why the accumulators are safe

The reference uses NumPy int64 storage to emulate the chosen hardware widths. Range checks and an analytical proof ensure that larger host storage does not hide an FPGA overflow.

For any permitted parameters and any uint8 image, conservative absolute bounds are:

    convolution: 2^31 + 9 * 255 * 32768
               = 2,222,686,208

    dense:       2^31 + 392 * 32767 * 32768
               = 423,041,433,600

    signed 40-bit positive limit
               = 549,755,813,887

Both bounds fit. They cover the bias and every accumulation prefix, so there is no intermediate wraparound even if positive and negative terms cancel later.

The [model manifest](../model/cnn-fixed-point/manifest.json) also computes tighter bounds from the exported weights. For this checkpoint, every possible pixel image gives an activation at most 1620 (about 6.328125 in real units), well below the 32767 saturation limit. Activation saturation is therefore impossible for this exported model, beyond just the images sampled in validation.

Saturation is nevertheless specified and tested, so the numerical contract remains clear. Accumulator overflow is an error, never an intended wraparound behavior.

## Measured validation results

The evaluated integer model was loaded back from the actual exported .mem files.

| Measure | Floating-point CNN | Fixed-point CNN |
|---|---:|---:|
| Validation images | 57,612 | 57,612 |
| Correct predictions | 56,151 | 56,154 |
| Accuracy | 97.464070% | 97.469277% |
| Macro recall | 97.010720% | 97.014347% |

Only 8 predictions differ: agreement is **99.986114%**. Quantization fixes 5 floating-point errors, introduces 2 errors, and changes one wrong answer to another wrong answer. The net change is +0.005207 percentage points; treat this as preserved accuracy, not evidence of a general improvement.

Across 90,335,616 convolution activations, observed saturation count is zero. The largest observed activation integer is 1047. Mean absolute logit error is 0.009368 and maximum absolute error is 0.059006, measured after restoring scores to real units.

See [the results report](../results/fixed-point/README.md), [metrics](../results/fixed-point/metrics.json), and [per-class results](../results/fixed-point/per_class.csv). These are development validation results, not a final test score or an FPGA speed measurement.

## FPGA export

The [model directory](../model/cnn-fixed-point/README.md) contains:

| File | Words | Word width | Address order |
|---|---:|---:|---|
| conv_weight.mem | 72 | 16 bits | channel, kernel row, kernel column |
| conv_bias.mem | 8 | 32 bits | channel |
| fc_weight.mem | 6,272 | 16 bits | class, feature index |
| fc_bias.mem | 16 | 32 bits | class |

Each line is one fixed-width hexadecimal word, with no prefix. Signed values use two's complement. These files are suitable inputs to Verilog $readmemh; their paths must be available to the simulator and Vivado project.

Raw parameter storage is **12,784 bytes**, excluding memory padding and FPGA implementation overhead. A design that stores both full post-ReLU convolution and pooled feature maps needs another 3,920 bytes, plus 256 input bytes. A later RTL design may reuse or stream those buffers.

The generated fixed_point.vh supplies localparam widths and dimensions. Include it inside a module declaration. It specifies constants; it is not an inference module.

manifest.json records scales, ordering, checkpoint/data/source hashes, software versions, memory-file hashes, and bounds. Git attributes pin the published bundle's line endings and preserve the baseline's Windows source bytes, because hashes depend on exact bytes.

## Verification evidence

All **26 software tests pass**, including:

- Positive and negative half ties, overflow rejection, and activation clipping boundaries.
- Hand-calculated asymmetric convolution, bias scale, max pooling, and feature order.
- All-negative output scores and tied argmax.
- Random parameter extremes and maximum-width accumulator cases.
- Two's-complement encoding, memory round trips, and corrupted-file detection.

An independent scalar implementation uses explicit Python-integer address and MAC loops. It checks every partial sum. It matches the vectorized implementation at every stage for **24 exported test cases**: the first validation image of each class and eight synthetic patterns.

A fresh process then reloaded the memory files, re-quantized the source checkpoint for comparison, checked all vector files, and recomputed the full validation set:

- 57,612 predictions matched.
- All 921,792 integer scores matched through their SHA-256 digest.
- 85,080 intermediate/output words across the 24 cases matched.
- No parameter, scalar-reference, or memory mismatch occurred.

See [verification.json](../results/fixed-point/verification.json) and [the vector guide](../test_vectors/cnn-fixed-point/README.md). This validates the software arithmetic contract; it does not substitute for the next RTL simulation checks.

## Reproduce this step

Use the Python environment from Step 1. No retraining is required.

From the repository root, verify the published bundle:

    $HexSightPython = (Get-Content local.settings.json | ConvertFrom-Json).python
    & $HexSightPython -m unittest discover -s python/tests -v
    & $HexSightPython python/verify_fixed_point.py

On another machine, set $HexSightPython to the Python executable in your environment. Pass --data-dir 'C:\path\to\emnist-data' when needed. The official training files and audited preprocessing cache are checked; official test images are never loaded.

Export a reproduction to three new directories:

    & $HexSightPython python/quantize.py --model-dir results/local/reproduced-model --output-dir results/local/reproduced-report --vectors-dir results/local/reproduced-vectors
    & $HexSightPython python/verify_fixed_point.py --model-dir results/local/reproduced-model --results-dir results/local/reproduced-report --vectors-dir results/local/reproduced-vectors

The exporter refuses to overwrite nonempty output directories. Both commands default to results/cnn-8filters as the floating-point source; --run-dir selects another compatible, fully verified CNN run. Validation predictions are checked against that run's saved predictions.

## Continue to Step 4: Verilog inference

The [Step 4 walkthrough](step-04.md) implements and simulates signed multiplication/accumulation, convolution addressing, activation rescaling, pooling, dense inference, and argmax.

Use the exported vectors to compare each stage exactly, beginning with convolution accumulators rather than only comparing the final predicted character. After software/RTL agreement, synthesize for the Nexys A7 and measure resource use and timing. UART and display integration follow afterward.

The milestone branches are step-02-cnn-baseline for training and step-03-fixed-point for this conversion.
