# HexSight FPGA

Handwritten hexadecimal recognition with a small neural network, from Python training to verified Verilog inference on the Nexys A7-100T.

Recognize one handwritten character from **0-9 and uppercase A-F**. A laptop prepares and sends the pixels; the FPGA will compute the network output and display the prediction. On a seven-segment display, B and D use the readable glyphs b and d.

## Current status

**Step 4 complete: verified Verilog CNN inference.** RTL simulation matches all **88,625 intermediate/output values** across 25 inferences, including all 24 frozen reference cases and a repeat. The core uses one shared signed MAC and completes each inference in **69,433 clock cycles**. Synthesis uses **1 DSP and 5.5 block-RAM tiles**; internal routed timing passes at 100 MHz. Full board/port timing remains open.

See [the RTL walkthrough](docs/step-04.md), [RTL verification and synthesis evidence](results/rtl-inference/README.md), and [the fixed-point contract](docs/step-03.md).

```text
16x16 grayscale -> 3x3 Conv (8 filters) -> ReLU -> 2x2 MaxPool
               -> Flatten (392) -> Linear (16 scores) -> argmax
```

UART/display integration and complete board timing are next. The official test set remains unused, and the board has not yet been programmed. No CPU speedup is claimed.

This is supervised classification: labeled images teach the network which character to predict. Training runs in PyTorch on the CPU. The FPGA will perform inference using exported, fixed-point weights.

## Milestones

1. **Environment and dataset:** verify labels, orientation, preprocessing, and data partitions.
2. **CNN training and verification:** train the convolutional model, reproduce predictions, and compare against the earlier MLP.
3. **Integer reference:** specify rounding and overflow, quantize weights, verify accuracy, and export memory files and intermediate vectors.
4. **Verilog inference:** implement the datapath, compare every intermediate stage, and assess standalone resources/internal timing.
5. **Board integration:** connect UART, inference, and the seven-segment display.
6. **Evaluation and release:** test held-out images, benchmark fairly, and prepare the course demo.

We finish and review each checkpoint before moving to the next one.

## Setup

Use Python 3.12 and a virtual environment. The requirements select CPU-only PyTorch wheels. The tested direct dependencies are pinned; `requirements-lock.txt` also records transitive versions from the verified Windows environment.

PowerShell, from the repository root:

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe -m unittest discover -s python/tests -v
```

An environment and dataset can live on another disk. Set these variables to those locations and use the same commands:

```powershell
$HexSightPython = 'C:\path\to\environment\Scripts\python.exe'
$HexSightData = 'C:\path\to\emnist-data'
& $HexSightPython python/prepare_data.py --download --data-dir $HexSightData
```

The dataset downloader uses NIST's official archive and torchvision's checksum validation. It downloads the archive containing all EMNIST splits; allow several GB of disk space for extraction. Only ByClass is used here. Omit `--download` when the files are already present.

Do not commit raw datasets or virtual environments. The supplied course handouts remain local references.

## Step 1: inspect the dataset

```powershell
& .\.venv\Scripts\python.exe python/prepare_data.py --download
```

This creates:

- [Sample preview](results/data_audit/samples.png): stored pixels, upright pixels, and the proposed 16x16 input.
- [Manifest](results/data_audit/manifest.json): class counts, preprocessing settings, software versions, and SHA-256 source hashes.
- [Class mapping](results/data_audit/class_mapping.json): characters mapped to output indices 0-15.
- `results/data_audit/split_indices.npz`: deterministic training, validation, and test indices.

The preview uses **training images only**. Training and validation indices refer to the official training split; test indices refer to the separate official test split. Their numeric indices may overlap because they address different files.

Verified on November 5, 2026:

| Partition | Images | Purpose |
|---|---:|---|
| Training | 326,480 | Fit the network |
| Validation | 57,612 | Choose checkpoints and model settings |
| Test | 64,437 | Final evaluation after model selection |

The classes are imbalanced. The training loss uses inverse training-frequency weights. Report both accuracy and per-class recall; overall accuracy alone can conceal weak letter recognition.

See [the first-step walkthrough](docs/step-01.md) for preprocessing details and what to inspect.

## Step 2: train and verify the CNN

Use the CNN configuration and a new directory for each training experiment:

```powershell
& .\.venv\Scripts\python.exe python/train.py --config configs/cnn.json --smoke --run-dir results/local/cnn-smoke-new
& .\.venv\Scripts\python.exe python/train.py --config configs/cnn.json --run-dir results/cnn-reproduced --threads 1
& .\.venv\Scripts\python.exe python/verify_checkpoint.py --run-dir results/cnn-reproduced
```

The CNN uses eight 3x3 filters, ReLU, 2x2 max pooling, and a 16-output dense layer. It has 6,368 parameters.
The selected checkpoint is epoch 46 of 50, with 97.46% validation accuracy and 97.01% macro recall.
See [Step 2](docs/step-02.md) for training, checkpoint selection, error analysis, and reproduction instructions.

Pass `--data-dir` when the dataset is outside the repository. On this workstation, the ignored `local.settings.json` records the dataset location and existing Python interpreter.

## CNN comparison (part of Step 2)

```powershell
& .\.venv\Scripts\python.exe python/compare_models.py --cnn-dir results/cnn-reproduced --output-dir results/comparison-reproduced
```

The earlier MLP results remain available as comparison evidence. The CNN reduces 8-to-B validation errors from 233 to 105, and 0-to-D errors from 232 to 190.
It needs about 2.34 times as many multiply-accumulate operations as the MLP, plus pooling and intermediate feature storage.

See [The comparison supplement](docs/cnn-comparison.md) for the comparison controls and FPGA implications, and [the comparison report](results/model-comparison/README.md) for measured results.

## Step 3: fixed-point conversion and export

The numerical contract uses 16-bit weights with 12 fractional bits, 16-bit activations with 8 fractional bits, 32-bit biases, and 40-bit accumulators. Integer inference uses the same topology and pixel normalization as the trained CNN.

Verify the published exports without retraining:

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s python/tests -v
& .\.venv\Scripts\python.exe python/verify_fixed_point.py
```

All 26 tests pass. Verification covers the exported parameters, 57,612 predictions, 921,792 integer scores, and intermediate values for 24 independent scalar-checked cases. Parameter storage is 12,784 bytes.

See [Step 3](docs/step-03.md) for equations, overflow proofs, memory addresses, reproduction commands, and next steps for RTL. No FPGA speedup or synthesis result is claimed yet.

## Step 4: simulate and synthesize the RTL

```powershell
& .\.venv\Scripts\python.exe python/run_rtl.py --results-dir results/rtl-reproduced
```

The runner uses Vivado from local.settings.json or --vivado. It stages builds on a local drive, checks the exported weights/vectors, runs arithmetic and full-CNN testbenches, and synthesizes/routes the standalone core. Raw EMNIST data is not needed.

The measured 69,433-cycle latency corresponds to 694.33 microseconds of core computation at 100 MHz, excluding input transfer. Standalone port timing is not closed; the [report](results/rtl-inference/README.md) distinguishes internal timing from the boundary paths that require board integration.

See [Step 4](docs/step-04.md) for the input handshake, reset behavior, state-machine schedule, verification coverage, and reproduction details.

## Initial preprocessing contract

1. Select the requested class names from EMNIST ByClass.
2. Transpose raw EMNIST images once to make them upright.
3. Find foreground pixels with intensity at least 16; add one pixel of bounding-box padding.
4. Resize with bilinear interpolation, preserving aspect ratio, so the longest dimension is 12 pixels.
5. Center on a black 16x16 canvas; keep unsigned 8-bit pixel values.
6. At model input, convert to float32 and divide by **256**, matching the planned fixed-point representation.

Blank images remain blank in the preprocessing function. The future drawing interface must reject blank input; the classifier itself has only 16 classes and no unknown class.

The resize and model size are initial choices, to be assessed on validation data before freezing the hardware interface.

## Repository layout

```text
configs/        Experiment configuration
python/         Dataset tools, training, fixed-point inference, and RTL runner
python/tests/   Software correctness checks
rtl/            Synthesizable CNN, shared MAC, activation, and argmax
tb/             Self-checking arithmetic and full-CNN SystemVerilog testbenches
constraints/    Standalone core timing; board pin constraints upcoming
scripts/        Vivado core synthesis and routing Tcl
model/          Verified fixed-point weights, biases, and Verilog constants
test_vectors/   Input and intermediate memory files for RTL verification
results/        Small, reproducible evidence and reports
docs/           Step-by-step explanations
```

## Development workflow

Each completed step is committed and pushed on its own milestone branch. See [the branch workflow](docs/git-workflow.md) for branch names, dependency order, and publication checks.

## Hardware target and evaluation

Target: Nexys A7-100T, Artix-7 `xc7a100tcsg324-1`, Vivado 2024.1, 100 MHz board clock. Pin assignments and timing constraints will be verified before programming the board.

Measure FPGA computation and laptop-to-board latency separately. A small FPGA design is not automatically faster than optimized CPU code; UART transfer can dominate total latency.

## Sources

- [NIST EMNIST dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)
- [Torchvision EMNIST loader](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.EMNIST.html)
- [Torchvision source and class-name mapping](https://docs.pytorch.org/vision/main/_modules/torchvision/datasets/mnist.html)
- [EMNIST orientation note](https://www.tensorflow.org/datasets/catalog/emnist)
- [Official PyTorch version/install pairs](https://pytorch.org/get-started/previous-versions/)

Project code is covered by the MIT license in [LICENSE](LICENSE). That license does not relicense third-party datasets or course handouts.
