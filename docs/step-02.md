# Step 2: train the CNN on the CPU

Updated November 5, 2026. This walkthrough uses the project's active convolutional neural network (CNN).
The trained checkpoint has been independently verified. The earlier MLP is retained as a comparison experiment; [The comparison supplement](cnn-comparison.md) records that comparison.

## What the neural network does

Start with the 16x16 grayscale images prepared in Step 1. Convert pixel values to float32 and divide by 256.
The cache stores each image as 256 values in row-major order; the CNN reshapes them back into a single-channel image.

```text
16x16 grayscale image
  -> 3x3 convolution, 8 filters, stride 1, no padding
  -> ReLU                         (8 x 14 x 14)
  -> 2x2 max pooling, stride 2     (8 x 7 x 7)
  -> flatten                      (392 values)
  -> fully connected              (16 raw scores)
  -> argmax                       (0-9 or A-F)
```

The convolution slides eight learned filters across the image. Each filter uses the same nine weights at every position to detect local patterns.
ReLU replaces negative activations with zero. Max pooling keeps the largest activation from each non-overlapping 2x2 region.
The dense layer combines the 392 pooled features into one score per character. The largest score determines the prediction.

The network has **6,344 weights and 24 biases: 6,368 trainable parameters**:

| Layer | Weights | Biases | Output shape per image |
|---|---:|---:|---|
| 3x3 convolution, 8 filters | 72 | 8 | 8 x 14 x 14 |
| ReLU | 0 | 0 | 8 x 14 x 14 |
| 2x2 max pooling | 0 | 0 | 8 x 7 x 7 |
| Fully connected | 6,272 | 16 | 16 scores |

PyTorch Conv2d uses cross-correlation, so the filter is not flipped.
Features flatten in channel, row, column order: all 49 pooled values from channel 0, then channel 1, and so on.
The future integer reference and Verilog implementation must use this same ordering.

## How training works

This is supervised classification using the labeled EMNIST ByClass subset for 0-9 and uppercase A-F.

For every batch of 128 training images:

1. The CNN computes its 16 output scores.
2. Class-weighted cross-entropy measures prediction error.
3. Backpropagation computes gradients for convolution and dense-layer weights and biases.
4. Adam updates those parameters, using a learning rate of 0.001.

Do not add softmax before cross-entropy; the loss consumes raw scores.
For the predicted class, argmax can operate directly on those scores.

The training partition contains 326,480 images, and the validation partition contains 57,612.
The separate official test subset contains 64,437 images and remains unused for model selection.

Class weights come from the training partition only:

```text
weight[class] = training_count / (16 * training_examples_in_class)
```

This equalizes summed training weight across classes, giving less frequent letters a stronger influence.
The validation loss uses these same training-derived weights.
For correct reporting across batches, sum weighted losses and divide by the sum of target weights across the entire evaluated partition.

## Choosing a checkpoint

Evaluate the fixed validation split after every epoch. Save a checkpoint whenever weighted validation cross-entropy improves.
Stop after eight consecutive epochs without improvement, or after 50 epochs.

This run reached the 50-epoch limit. **Epoch 46** had the lowest validation loss and is the saved model.
The selection rule is based on loss, rather than choosing the epoch with the highest raw accuracy.

The plotted training loss is measured during parameter updates. Validation loss is measured after each epoch.
The curves therefore differ in measurement timing as well as the images used.

## Results

| Measure | Selected CNN checkpoint |
|---|---:|
| Validation images | 57,612 |
| Correct predictions | 56,151 |
| Validation accuracy | 97.46% |
| Macro recall | 97.01% |
| Weighted validation cross-entropy | 0.10063 |
| Selected epoch | 46 of 50 |
| Training/validation loop time | 392.3 seconds |
| Fresh-process prediction mismatches | 0 of 57,612 |

Macro recall averages recognition rate equally across the 16 classes.
Accuracy gives every image equal weight, so common digits influence it more strongly.

Runtime includes epoch training, validation, and checkpoint/history writes with one CPU thread and cached preprocessed images.
It excludes the initial dataset preprocessing and final plots. It is not an inference benchmark or an FPGA speed measurement.

The held-out test set has **not** been evaluated. This checkpoint has not yet been quantized or run on hardware.

## Understand the mistakes

The two largest confusion counts are:

- 190 true 0 images predicted as D.
- 105 true 8 images predicted as B.

B has 94.16% recall and 76.64% precision in this imbalanced validation set.
D has 94.01% recall and 73.49% precision.
Recall measures how many actual examples of a character are recognized; precision measures how many predictions of that character are correct.
These weaknesses are important even with high overall accuracy.

See the [CNN confusion matrix](../results/cnn-8filters/confusion_matrix.png) and [example mistakes](../results/cnn-8filters/validation_errors.png).
The gallery shows the first mistake for each true class in frozen validation order.
[The comparison supplement](cnn-comparison.md) compares these results with the MLP on the same images.

## Reproduce the run

From the repository root, using the environment created in Step 1:

```powershell
$HexSightPython = '.\.venv\Scripts\python.exe'
& $HexSightPython -m pip install -r requirements-lock.txt
& $HexSightPython -m unittest discover -s python/tests -v
& $HexSightPython python/train.py --config configs/cnn.json --smoke --run-dir results/local/cnn-smoke-new
& $HexSightPython python/train.py --config configs/cnn.json --run-dir results/cnn-reproduced --threads 1
& $HexSightPython python/verify_checkpoint.py --run-dir results/cnn-reproduced
```

The default train.py configuration is configs/cnn.json; specifying it explicitly makes the experiment clear.

For this workstation's existing environment:

```powershell
$HexSightPython = (Get-Content local.settings.json | ConvertFrom-Json).python
& $HexSightPython python/verify_checkpoint.py --run-dir results/cnn-8filters
```

Training and verification read the dataset path from local.settings.json when present.
On another machine, pass `--data-dir 'C:\path\to\emnist-data'`, or use the repository's ignored data directory.
Choose a new output directory for each training run; the script refuses to overwrite an existing experiment.

The cache identity includes source hashes, saved partitions, preprocessing settings, and preprocessing code.
Reproduction uses seed 311, deterministic PyTorch operations, pinned dependencies, and a recorded CPU thread count.
Exact numerical equality across different hardware or software versions is not promised.

## Artifacts and checks

- [CNN configuration](../configs/cnn.json).
- [Training implementation](../python/train.py) and [network definition](../python/hexsight/model.py).
- [Saved CNN](../results/cnn-8filters/best.pt).
- [Learning curves](../results/cnn-8filters/learning_curves.png).
- [Metrics](../results/cnn-8filters/metrics.json) and [per-class CSV](../results/cnn-8filters/per_class.csv).
- [Run provenance](../results/cnn-8filters/run.json).
- [Independent verification](../results/cnn-8filters/verification.json).

All 15 software tests passed in the verified CNN milestone.
They cover preprocessing, class weighting, metric calculations, hand-computed convolution and pooling, feature ordering, learnable toy problems, and checkpoint reloading.
The short smoke run reduced validation loss, and a fresh process reproduced every prediction from the selected full CNN checkpoint.

The checkpoint contains a state_dict and configuration. Its loader explicitly uses weights_only=True.
The current source, CNN checkpoint, and reports are published on the step-02-cnn-baseline branch, which consolidates the training work.

## Next checkpoint

Review [the CNN comparison and verification supplement](cnn-comparison.md).
The next implementation milestone is fixed-point CNN inference: quantize the weights, specify convolution accumulation, ReLU, pooling, flattening order, and the 392-input dense layer, then verify rounding and accumulator ranges.
That integer Python reference will define the numerical behavior required from Verilog.

## References

- [PyTorch Conv2d and cross-correlation](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Conv2d.html)
- [PyTorch MaxPool2d](https://docs.pytorch.org/docs/2.9/generated/torch.nn.MaxPool2d.html)
- [PyTorch cross-entropy and class weighting](https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html)
- [PyTorch reproducibility](https://docs.pytorch.org/docs/2.9/notes/randomness.html)
