# HexSight CNN baseline

Completed November 5, 2026. Validation results only; the final test set remains unused.

Architecture: 16x16 image -> Conv2d(1, 8, 3), no padding -> ReLU -> MaxPool2d(2) -> Flatten(392) -> Linear(392, 16).

| Measure | Value |
|---|---:|
| Validation accuracy | 97.4641% |
| Macro recall | 97.0107% |
| Macro F1 | 95.5244% |
| Weighted validation loss | 0.100635 |
| Correct validation predictions | 56,151 / 57,612 |
| Best epoch by weighted validation loss | 46 |
| Epochs completed | 50 |
| Parameters | 6,368 |
| Training/validation loop seconds | 392.286 |

Training reached the 50-epoch limit. The saved checkpoint is epoch 46, selected by validation loss rather than maximum accuracy.

## Per-class results

| Character | Images | Recall | Precision |
|---|---:|---:|---:|
| 0 | 5,188 | 95.14% | 98.42% |
| 1 | 5,756 | 99.50% | 99.27% |
| 2 | 5,130 | 97.64% | 98.45% |
| 3 | 5,271 | 97.38% | 98.64% |
| 4 | 5,030 | 98.07% | 98.35% |
| 5 | 4,712 | 97.79% | 98.42% |
| 6 | 5,135 | 98.27% | 99.39% |
| 7 | 5,363 | 98.56% | 98.67% |
| 8 | 5,092 | 95.03% | 98.13% |
| 9 | 5,077 | 97.93% | 97.74% |
| A | 961 | 97.81% | 93.25% |
| B | 582 | 94.16% | 76.64% |
| C | 1,514 | 98.02% | 93.75% |
| D | 684 | 94.01% | 73.49% |
| E | 740 | 95.54% | 95.15% |
| F | 1,377 | 97.31% | 91.66% |

## Artifacts

- [Checkpoint](best.pt) and [run provenance](run.json).
- [Learning curves](learning_curves.png), [confusion matrix](confusion_matrix.png), and [example errors](validation_errors.png).
- [Metrics](metrics.json), [per-class CSV](per_class.csv), and [epoch history](history.csv).
- [Validation predictions](validation_predictions.npz) and [fresh-process verification](verification.json).

A fresh process reproduced all 57,612 validation predictions with zero mismatches.
The error gallery uses the first error per true class in saved validation order.

Runtime includes epoch training, validation, and checkpoint/history writes with one CPU thread and cached preprocessing. It is not an inference or FPGA speed benchmark.

Checkpoint SHA-256: `109556fe96c39b8cdbf54eaae6a1b42bf0c344938fc8133f3ed0aeea2c6591b6`.

See [Step 3](../../docs/step-03.md) for the architecture, reproduction commands, and FPGA implications.
