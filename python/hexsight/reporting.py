"""Standalone plots for a completed validation run; no test data is used."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_results(history, metrics, pixels, predictions, targets, classes, best_epoch, run_dir):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for key, label in (("train_weighted_loss", "Training (during updates)"),
                       ("val_weighted_loss", "Validation")):
        axes[0].plot(epochs, [row[key] for row in history], label=label)
    axes[0].set(xlabel="Epoch", ylabel="Class-weighted cross-entropy", title="Learning progress")
    axes[0].legend()
    axes[1].plot(epochs, [100 * row["val_accuracy"] for row in history], label="Validation accuracy")
    axes[1].plot(epochs, [100 * row["val_macro_recall"] for row in history], label="Validation macro recall")
    axes[1].set(xlabel="Epoch", ylabel="Percent", title="Recognition quality")
    axes[1].legend()
    for axis in axes:
        axis.axvline(best_epoch, color="#64748b", ls="--", lw=1, label="Saved checkpoint")
        axis.grid(alpha=0.2)
    fig.suptitle(f"HexSight CPU baseline | best validation loss at epoch {best_epoch}")
    fig.savefig(run_dir / "learning_curves.png", dpi=160)
    plt.close(fig)

    matrix = np.asarray(metrics["confusion_matrix"])
    normalized = matrix / matrix.sum(axis=1, keepdims=True) * 100
    fig, axis = plt.subplots(figsize=(10, 9), layout="constrained")
    plot = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=100)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if normalized[i, j] >= 0.5:
                axis.text(j, i, f"{normalized[i, j]:.0f}", ha="center", va="center",
                          fontsize=8, color="white" if normalized[i, j] > 55 else "#132238")
    axis.set(xticks=range(len(classes)), xticklabels=list(classes),
             yticks=range(len(classes)), yticklabels=list(classes),
             xlabel="Predicted character", ylabel="True character",
             title="Validation confusion matrix | percent within each true class")
    fig.colorbar(plot, ax=axis, label="Percent of true class", shrink=0.8)
    fig.savefig(run_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    # First error per true class, in frozen validation order; not cherry-picked.
    chosen = []
    for label in range(len(classes)):
        errors = np.flatnonzero((targets == label) & (predictions != label))
        if len(errors):
            chosen.append(int(errors[0]))
    fig, axes = plt.subplots(4, 4, figsize=(8, 8), layout="constrained")
    side = int(np.sqrt(pixels.shape[1]))
    for axis in axes.flat:
        axis.axis("off")
    for axis, index in zip(axes.flat, chosen):
        axis.imshow(pixels[index].reshape(side, side), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axis.set_title(f"True {classes[targets[index]]} / predicted {classes[predictions[index]]}")
    fig.suptitle("Validation errors | first error per class in saved split order")
    fig.savefig(run_dir / "validation_errors.png", dpi=160)
    plt.close(fig)
