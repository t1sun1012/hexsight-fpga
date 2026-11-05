# CNN integer reference vectors

There are 24 cases:

- The first image of each of the 16 classes in the frozen validation order.
- Blank, full-white, checkerboard, two corner impulses, horizontal and vertical ramps, and random uint8 pixels with seed 311.

Synthetic patterns are arithmetic tests. Their predictions are not meaningful recognition labels; the future drawing UI should reject blank input.

Each .mem file concatenates all cases in manifest order. Within a case, values flatten in channel, row, column order where applicable. A case starts at:

    word_address = case_index * words_per_case

| File | Words per case | Width | Interpretation |
|---|---:|---:|---|
| pixels.mem | 256 | unsigned 8 | Row-major input bytes; fraction bits 8 |
| conv_acc.mem | 1,568 | signed 40 | Before ReLU/rescaling; fraction bits 20 |
| relu.mem | 1,568 | signed 16 | Rounded, clamped convolution output; fraction bits 8 |
| pool.mem | 392 | signed 16 | Pooled/flattened features; fraction bits 8 |
| logits.mem | 16 | signed 40 | Dense scores; fraction bits 20 |
| predictions.mem | 1 | unsigned 4 | Class index 0-15 |

Each line contains one fixed-width hex word. Signed values use two's complement. All files use LF newlines. manifest.json includes dimensions, source indices, labels where applicable, expected predictions, and hashes binding the vectors to the exported model.

Both the vectorized NumPy integer reference and the independent scalar Python-integer implementation match all **85,080 intermediate/output words**. The separate 6,144 input bytes also round-trip exactly through memory export.

Next, the Verilog testbench should load these files and compare each stage bit for bit. Begin with convolution accumulators, then activation rounding, pooling, dense scores, and the predicted class. This will distinguish numerical errors from address-order errors.

Run python/verify_fixed_point.py from the repository root with the project Python environment to recheck the published vectors and full validation predictions.
