# CNN versus MLP

Comparison completed November 5, 2026. Both models were trained with the same settings and evaluated on the same 57,612 validation images.

| Measure | MLP | CNN |
|---|---:|---:|
| Validation accuracy | 95.37% | 97.46% |
| Macro recall | 94.64% | 97.01% |
| Macro F1 | 91.90% | 95.52% |
| Weighted validation loss | 0.18539 | 0.10063 |
| Selected epoch | 47 | 46 |
| Parameters | 8,752 | 6,368 |
| Estimated multiply-accumulates per image | 8,704 | 20,384 |
| Training/validation loop time (seconds) | 61.7 | 392.3 |

The CNN improves accuracy by **2.10 percentage points** and macro recall by **2.37 percentage points**.
It corrects 1,743 examples the MLP gets wrong, while getting 534 examples wrong that the MLP gets right.

## Common confusions

| True -> predicted | MLP errors | CNN errors |
|---|---:|---:|
| 8 -> B | 233 | 105 |
| 0 -> D | 232 | 190 |

B recall improves from 87.80% to 94.16%; B precision improves from 54.89% to 76.64%.
D recall improves from 92.25% to 94.01%; D precision improves from 64.98% to 73.49%.
B and D still have lower precision than most other classes.

## Interpretation

Use the CNN as the active model for the next fixed-point implementation milestone.
It has fewer parameters, but its convolution reuses weights at many locations: the arithmetic count is about 2.34 times the MLP's. FPGA latency and resource usage still require implementation and measurement.
This is a single-seed validation comparison. No official test-set score or FPGA speedup is claimed.
Both runs reached 50 epochs and used the checkpoint with lowest weighted validation loss. Runtime includes training, validation, and checkpoint/history writes; it excludes initial preprocessing and final plots.

## Evidence

- [Per-class comparison plot](per_class_comparison.png) and [CSV](per_class_comparison.csv).
- [Machine-readable comparison](comparison.json).
- [MLP report](../baseline-256-32-16/README.md) and [CNN report](../cnn-8filters/README.md).
- [Reproducible comparison command](../../python/compare_models.py).

The command checks matching source/cache/split hashes, training settings, class weights, versions, thread count, and exact validation indices. Both checkpoints reproduced all saved predictions in separate processes.
