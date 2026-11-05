# Step 3: compare and verify the CNN

[Step 2](step-02.md) is the main CNN training walkthrough. This document records the controlled comparison, verification, and FPGA implications.
The original MLP checkpoint remains available as the reference for comparison.

## Architecture

```text
16x16 grayscale image
  -> 3x3 convolution, 8 output channels, stride 1, no padding
  -> ReLU                         (8 x 14 x 14)
  -> 2x2 max pooling, stride 2     (8 x 7 x 7)
  -> flatten                      (392 values)
  -> fully connected              (16 raw scores)
  -> argmax                       (0-9 or A-F)
```

The existing cache still stores row-major flattened input images. The CNN reshapes each input back to a single-channel image internally.
Pixel normalization remains pixel / 256.

The convolution shares each filter's nine weights across the image.
ReLU clips negative activations to zero. Max pooling keeps the largest activation in each non-overlapping 2x2 region.
The dense layer combines the resulting 392 features into the 16 class scores.

PyTorch's Conv2d performs cross-correlation: it does **not** flip the filter.
The feature layout is channel, row, column. The dense layer therefore receives all 49 pooled values from channel 0, then channel 1, and so on.
These details must match the later Python integer reference and Verilog implementation.

## Controlled comparison

Both experiments use the same:

- 326,480 training and 57,612 validation images, in the saved partitions.
- Corrected orientation, crop, 16x16 resize, and pixel normalization.
- Seed 311, Adam learning rate 0.001, and batch size 128.
- Inverse-frequency training class weights.
- Maximum 50 epochs and early-stopping patience of eight epochs.
- Selection by lowest weighted validation cross-entropy.
- CPU execution with one PyTorch thread and the same dependency versions.

The comparison command checks the source, split, and cache hashes; settings; class weights; and original validation indices.
The final test split is not used for training, checkpoint selection, or architecture comparison.
This is a single-seed development comparison, not a statistical claim about all training runs.

## Results

Completed November 5, 2026. The selected CNN checkpoint (epoch 46) achieved **97.46% validation accuracy** and **97.01% macro recall**.
The MLP reference achieved 95.37% and 94.64%, respectively.
Both completed 50 epochs and were selected by minimum weighted validation loss.

All 15 tests pass. The CNN checkpoint reproduced all 57,612 saved validation predictions in a separate process with zero mismatches.
The training/validation loop took 392.3 seconds with cached input data and one CPU thread.

See [the measured comparison](../results/model-comparison/README.md), [per-class plot](../results/model-comparison/per_class_comparison.png), and [CNN run report](../results/cnn-8filters/README.md).

The CNN training command writes its selected checkpoint, learning curves, confusion matrix, per-class metrics, predictions, and provenance.
A separate verification command reloads the checkpoint and recomputes every saved validation prediction.

## Cost to implement on the FPGA

| Arithmetic/storage quantity | MLP | CNN |
|---|---:|---:|
| Learned weights | 8,704 | 6,344 |
| Learned biases | 48 | 24 |
| Total parameters | 8,752 | 6,368 |
| Multiply-accumulates per image | 8,704 | 20,384 |
| Max-pooling comparisons per image | 0 | 1,176 |

These are arithmetic counts, not synthesis results or measured latency.
The CNN has fewer parameters but about 2.34 times as many multiply-accumulates.
It also needs convolution addressing, intermediate feature storage, and max-pooling control.

The original project plan described an MLP. The next fixed-point milestone must now specify convolution accumulation, ReLU, pooling, channel-major flattening, and a 392-input dense layer.
It must check accumulator ranges and rounding for this architecture instead of reusing the old dense-layer dimensions.

## Run it

From the repository root:

```powershell
$HexSightPython = '.\.venv\Scripts\python.exe'
& $HexSightPython -m unittest discover -s python/tests -v
& $HexSightPython python/train.py --config configs/cnn.json --smoke --run-dir results/local/cnn-smoke-new
& $HexSightPython python/train.py --config configs/cnn.json --run-dir results/cnn-reproduced --threads 1
& $HexSightPython python/verify_checkpoint.py --run-dir results/cnn-reproduced
& $HexSightPython python/compare_models.py --cnn-dir results/cnn-reproduced --output-dir results/comparison-reproduced
```

For the environment already configured on this workstation:

```powershell
$HexSightPython = (Get-Content local.settings.json | ConvertFrom-Json).python
```

The default configuration for train.py is now configs/cnn.json.
To reproduce the old MLP experiment, pass `--config configs/baseline.json` explicitly.
The checkpoint loader recognizes older configurations without a model type as MLP, preserving compatibility.

Data location can be supplied with `--data-dir`; otherwise the training and verification commands read the ignored local.settings.json or use the repository data directory.
Use a new run directory for every training experiment.

## Checks

The CNN-specific tests check hand-computed convolution and pooling values, kernel orientation, channel/spatial flattening order, architecture counts, learning a tiny spatial problem, checkpoint reload, and invalid input shapes.
The original preprocessing and MLP tests remain part of the suite.
The original full MLP checkpoint also continues to reproduce all 57,612 validation predictions.

## Next milestone

Implement and validate fixed-point **CNN** inference in Python, then export convolution and dense-layer weights for Verilog.
Neither the floating-point CNN nor the MLP has been run on the FPGA yet.

## References

- [PyTorch Conv2d and cross-correlation semantics](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Conv2d.html)
- [PyTorch MaxPool2d](https://docs.pytorch.org/docs/2.9/generated/torch.nn.MaxPool2d.html)
