# HexSight FPGA

Handwritten hexadecimal recognition with a small neural network, from Python training to verified Verilog inference on the Nexys A7-100T.

Recognize one handwritten character from **0-9 and uppercase A-F**. A laptop prepares and sends the pixels; the FPGA will compute the network output and display the prediction. On a seven-segment display, B and D use the readable glyphs b and d.

## Current status

**Step 1 complete: CPU environment and dataset audit.** Model training and FPGA implementation are the next milestones. No accuracy, hardware timing, or speedup results have been measured yet.

The initial candidate is a fully connected network:

```text
16x16 grayscale image -> 256 inputs -> 32 hidden units + ReLU -> 16 scores
```

This is supervised classification: labeled images teach the network which character to predict. Training will run in PyTorch on the CPU. The FPGA will perform inference using exported, fixed-point weights.

## Milestones

1. **Environment and dataset:** verify labels, orientation, preprocessing, and data partitions.
2. **Python baseline:** train the 256-32-16 model; measure validation accuracy and per-class recall.
3. **Integer reference:** specify rounding and overflow, quantize weights, and export memory files.
4. **Verilog inference:** compare intermediate results and predictions against the integer reference.
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

The classes are imbalanced. The planned training loss uses inverse training-frequency weights. Report both accuracy and per-class recall; overall accuracy alone can conceal weak letter recognition.

See [the first-step walkthrough](docs/step-01.md) for preprocessing details and what to inspect.

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
python/         Dataset tools and, later, training/reference code
python/tests/   Software correctness checks
rtl/            Synthesizable Verilog (upcoming)
tb/             Verilog testbenches (upcoming)
constraints/    Board pin and timing constraints (upcoming)
model/          Exported inference parameters (upcoming)
test_vectors/   Python-to-RTL verification cases (upcoming)
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
