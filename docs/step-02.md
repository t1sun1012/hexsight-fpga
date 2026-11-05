# Step 2: train the CPU baseline

Completed November 5, 2026. The first floating-point model is trained and its saved checkpoint has been independently verified.

## What the neural network does

Each preprocessed 16x16 image contains 256 pixel values. Flatten the image in row-major order, convert to float32, and divide by 256.

The first linear layer computes 32 weighted sums, each with a learned bias.
ReLU replaces negative results with zero. The second linear layer converts those 32 values to 16 scores, one per character.
The largest score is the prediction.

```text
256 pixels -> Linear(256, 32) -> ReLU -> Linear(32, 16) -> argmax
```

The model has 8,704 weights and 48 biases: **8,752 trainable parameters**.
There is no convolution, clustering, or reinforcement learning in this baseline.

## How training works

For every batch of 128 labeled images:

1. The network computes its 16 output scores.
2. Class-weighted cross-entropy measures prediction error.
3. Backpropagation computes gradients for the weights and biases.
4. Adam updates those parameters, using a learning rate of 0.001.

Do not add softmax to the model before cross-entropy; the loss consumes raw scores.
The future FPGA only needs the largest score, so it will not need softmax either.

Class weights are derived from the training partition only:

```text
weight[class] = training_count / (16 * training_examples_in_class)
```

This makes the summed training weight equal across classes, giving less frequent letters a stronger influence.
The validation loss uses these same training-derived weights.
For correct reporting across batches, sum the weighted losses and divide by the sum of target weights over the entire partition.

## Choosing a checkpoint

Evaluate the fixed validation split after every epoch. Save a checkpoint whenever its weighted validation cross-entropy improves.
Stop after eight consecutive epochs without improvement, or after 50 epochs.

This run reached the 50-epoch limit. **Epoch 47** had the lowest validation loss and is the saved model.
The highest raw accuracy occurred at a different epoch; we keep the selection rule fixed instead of changing it after seeing the results.

The training loss in the learning curve is measured during parameter updates. Validation loss is measured after the epoch.
These differ in timing as well as data, so their curves are not a direct measurement of the same frozen model on two sets.

## Results

| Measure | Selected checkpoint |
|---|---:|
| Validation images | 57,612 |
| Validation accuracy | 95.37% |
| Macro recall | 94.64% |
| Weighted validation cross-entropy | 0.18539 |
| Selected epoch | 47 of 50 |
| Training/validation loop time | 61.7 seconds |
| Fresh-process prediction mismatches | 0 of 57,612 |

Macro recall averages the recognition rate for each character equally.
Accuracy gives every image equal weight, so common digits influence it more strongly.

The runtime includes the epoch loops, validation, and checkpoint/history writes on this CPU with one PyTorch thread, using cached preprocessed images.
It excludes the initial dataset preprocessing and final plot generation. It is not an inference benchmark or a comparison against FPGA hardware.

These results exceed the plan's provisional validation targets of 90% accuracy and 85% macro recall.
The held-out test set has **not** been evaluated, and this checkpoint has not yet been quantized or run on hardware.

## Understand the mistakes

The two largest error counts are:

- 233 true 8 images predicted as B.
- 232 true 0 images predicted as D.

B has 87.80% recall, but only 54.89% precision in this imbalanced validation set.
Recall asks how many actual B images are recognized. Precision asks how many B predictions are actually B.
D also has lower precision, 64.98%, despite 92.25% recall. Overall accuracy alone would hide these weaknesses.

See the [confusion matrix](../results/baseline-256-32-16/confusion_matrix.png) and [example mistakes](../results/baseline-256-32-16/validation_errors.png).
The error gallery shows the first mistake for each true class in the frozen validation order, rather than hand-selected examples.
Some inputs are visually ambiguous at 16x16. A larger input or changed class weighting could be compared later, but this baseline remains the reference for the next step.

## Reproduce the run

From the repository root, using the environment created in Step 1:

```powershell
$HexSightPython = '.\.venv\Scripts\python.exe'
& $HexSightPython -m pip install -r requirements-lock.txt
& $HexSightPython -m unittest discover -s python/tests -v
& $HexSightPython python/train.py --smoke --run-dir results/local/smoke-new
& $HexSightPython python/train.py --run-dir results/baseline-reproduced --threads 1
& $HexSightPython python/verify_checkpoint.py --run-dir results/baseline-reproduced
```

For this workstation's existing environment, set the interpreter using the ignored machine-local file:

```powershell
$HexSightPython = (Get-Content local.settings.json | ConvertFrom-Json).python
& $HexSightPython python/verify_checkpoint.py --run-dir results/baseline-256-32-16
```

Both commands read the dataset path from local.settings.json when present. On another machine, pass `--data-dir 'C:\path\to\emnist-data'`, or place the data under the repository's ignored data directory.
Choose a new output directory for each run; the script refuses to overwrite an existing experiment.

The cache records its identity from source hashes, the saved split, preprocessing settings, and preprocessing code.
Reproduction uses a fixed seed, deterministic PyTorch operations, pinned dependencies, and recorded CPU thread count.
Exact numerical equality across different hardware or software versions is not promised.

## Artifacts and checks

- [Training implementation](../python/train.py).
- [Network definition](../python/hexsight/model.py).
- [Saved model](../results/baseline-256-32-16/best.pt).
- [Learning curves](../results/baseline-256-32-16/learning_curves.png).
- [Machine-readable metrics](../results/baseline-256-32-16/metrics.json).
- [Per-class metrics](../results/baseline-256-32-16/per_class.csv).
- [Run provenance](../results/baseline-256-32-16/run.json).
- [Independent verification result](../results/baseline-256-32-16/verification.json).

All nine software tests pass. They include a small learnable toy problem, checkpoint reloading, class-weight semantics, and a known-answer confusion matrix.
The short smoke run also reduced validation loss on a balanced subset.
A fresh Python process then reproduced every prediction from the full selected checkpoint.

The checkpoint contains a state_dict and configuration. The loader explicitly uses weights_only=True.
FPGA inference and board programming remain future work. Source code, the checkpoint, and validation reports are published on the Step 2 milestone branch.

## Next checkpoint

Build the integer Python reference: quantize the trained weights, define exact signed arithmetic and rounding, and compare integer inference against this floating-point baseline on the same validation images.
That will give the Verilog implementation a precise numerical target.

## References

- [PyTorch cross-entropy and class weighting](https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html)
- [PyTorch reproducibility controls and limits](https://docs.pytorch.org/docs/2.9/notes/randomness.html)
- [Saving and loading PyTorch models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
