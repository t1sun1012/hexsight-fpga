# Step 1: understand and verify the inputs

Completed November 5, 2026.

The goal is to know exactly what an image and its label mean before training.
An orientation or label mistake can produce a misleading model and later make correct FPGA arithmetic look broken.

## What the labels mean

We use EMNIST **ByClass** because it provides separate uppercase and lowercase source labels.
We retain the class names `0123456789ABCDEF` and map them to output indices 0 through 15.
The script looks up source labels by name instead of assuming their numerical order.

An image labeled A should train output 10. An image labeled F should train output 15.
The network predicts a character class. The display decoder later chooses which physical segments to light.

Handwriting is variable: some source samples resemble lowercase forms or other characters even within an uppercase-labeled class.
Keep the source labels unchanged and report the resulting confusions when evaluating the model.

## Read the preview

Open [samples.png](../results/data_audit/samples.png). Each row is one character, with four samples.

Every group contains three views:

1. **Raw:** pixels as stored by EMNIST.
2. **Upright:** the raw image transposed across its diagonal.
3. **16x16:** the cropped, resized, and centered input proposed for the network.

Use asymmetric characters such as 2, 7, F, and E to check orientation.
Symmetric characters such as 0 can hide an orientation bug.
This correction is for EMNIST storage. A future upright drawing canvas must not be transposed again.

The preview has been visually reviewed. It confirms the transpose corrects the dataset orientation.
Some handwriting remains ambiguous, and resizing removes detail; validation will determine whether the small input is sufficient.

## Why divide by 256?

Pixels are integers from 0 to 255. We keep those bytes for transmission and interpret them as pixel / 256 in the neural network.
For example, a byte value of 128 represents 0.5.

This deliberate choice makes the input representation compatible with eight fractional bits in the planned integer reference.
The largest possible normalized input is 255/256, not 1.
Use the same convention in training, export, Python reference, and hardware.

## How the data is partitioned

The official training split supplies our training and validation partitions.
Within each retained class, shuffle with a fixed NumPy seed of 311 and reserve 15% for validation.
Per-class validation counts round half up to a whole image.

Store original indices, class counts, configuration, and hashes so later experiments can use the same examples.
The official test split stays separate. Its class counts are recorded, but no test images appear in the preview and no test scores have been calculated.

Current totals:

- Training: 326,480.
- Validation: 57,612.
- Test: 64,437.

The largest retained training class is 1 with 32,618 images; B has 3,296.
In the next step, calculate loss weights from the training partition only and inspect recall for each class.

## Files worth reading first

1. [Baseline configuration](../configs/baseline.json): the choices we can compare experimentally.
2. [Preprocessing](../python/hexsight/data.py): the exact operations applied to each image.
3. [Preparation script](../python/prepare_data.py): class filtering, partitioning, and evidence generation.
4. [Tests](../python/tests/test_data.py): checks for mapping, orientation, and accidental data overlap.

The five tests pass. CPU PyTorch import, matrix multiplication, and dependency compatibility also pass.

## Next checkpoint

Follow [Step 2](step-02.md) to train the compact CNN with eight 3x3 filters, ReLU, max pooling, and 16 output scores.
Start with a short training smoke run, then train with early stopping on validation data.
Save the best checkpoint, learning curves, and a validation confusion matrix.

The model is trained by supervised learning using cross-entropy and backpropagation; this is not clustering.
No neural network has been trained at this checkpoint.
