# HexSight floating-point baseline

Recorded November 5, 2026. Validation results only; final test set not evaluated.

| Measure | Value |
|---|---:|
| Validation accuracy | 95.3655% |
| Macro recall | 94.6415% |
| Macro F1 | 91.9011% |
| Best epoch (lowest weighted loss) | 47 |
| Epochs completed | 50 |
| Parameters | 8,752 |
| Training/validation loop seconds | 61.713 |

## Per-class recognition

| Character | Validation images | Recall | Precision |
|---|---:|---:|---:|
| 0 | 5,188 | 93.60% | 97.73% |
| 1 | 5,756 | 98.91% | 97.89% |
| 2 | 5,130 | 95.11% | 98.21% |
| 3 | 5,271 | 94.16% | 97.93% |
| 4 | 5,030 | 95.96% | 97.46% |
| 5 | 4,712 | 94.57% | 97.23% |
| 6 | 5,135 | 97.43% | 97.89% |
| 7 | 5,363 | 98.27% | 97.48% |
| 8 | 5,092 | 90.95% | 96.90% |
| 9 | 5,077 | 95.59% | 96.33% |
| A | 961 | 94.80% | 88.96% |
| B | 582 | 87.80% | 54.89% |
| C | 1,514 | 96.76% | 88.57% |
| D | 684 | 92.25% | 64.98% |
| E | 740 | 93.78% | 81.26% |
| F | 1,377 | 94.34% | 87.36% |

## Artifacts

- best.pt: selected PyTorch checkpoint, including configuration and class weights.
- run.json: training settings, versions, source hashes, and code hashes.
- metrics.json / per_class.csv: validation metrics for the selected checkpoint.
- history.csv: all epochs, including those after the selected checkpoint.
- validation_predictions.npz: official training-file indices, true labels, and predictions for validation.
- verification.json: fresh-process reproduction of all 57,612 predictions, with zero mismatches.
- learning_curves.png / confusion_matrix.png / validation_errors.png: inspected diagnostic plots.

Checkpoint SHA-256: `7dea434bf293fb7bedf583a0dcee2a5f9d7e4d8dfc7b8c05d4208c3bde02417d`.

This is the historical MLP reference. See [the CNN comparison supplement](../../docs/cnn-comparison.md) and [the model comparison](../model-comparison/README.md) for interpretation and limitations. To reproduce this MLP, run `python python/train.py --config configs/baseline.json --run-dir results/mlp-reproduced --threads 1` using the project Python environment.
