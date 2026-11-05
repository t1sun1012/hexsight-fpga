"""Classification metrics with true classes on rows and predictions on columns."""

import numpy as np
import torch


def inverse_frequency_weights(labels, num_classes):
    counts = torch.bincount(labels, minlength=num_classes)
    if torch.any(counts == 0):
        raise ValueError("Training must contain every class")
    return len(labels) / (num_classes * counts.to(torch.float32))


def classification_metrics(targets, predictions, classes):
    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    count = len(classes)
    if targets.shape != predictions.shape or targets.ndim != 1 or not len(targets):
        raise ValueError("Expected equally sized, nonempty label vectors")
    if min(targets.min(), predictions.min()) < 0 or max(targets.max(), predictions.max()) >= count:
        raise ValueError("Label outside the class mapping")
    matrix = np.bincount(targets * count + predictions, minlength=count * count).reshape(count, count)
    support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    if np.any(support == 0):
        raise ValueError("Evaluation must contain every class for macro recall")
    correct = matrix.diagonal()
    recall = correct / support
    precision = np.divide(correct, predicted, out=np.zeros(count, dtype=float), where=predicted != 0)
    f1 = np.divide(2 * recall * precision, recall + precision, out=np.zeros(count), where=recall + precision != 0)
    per_class = [
        {"character": character, "support": int(support[i]), "correct": int(correct[i]),
         "recall": float(recall[i]), "precision": float(precision[i]), "f1": float(f1[i])}
        for i, character in enumerate(classes)
    ]
    confusions = sorted(
        ({"true": classes[i], "predicted": classes[j], "count": int(matrix[i, j]),
          "fraction_of_true_class": float(matrix[i, j] / support[i])}
         for i in range(count) for j in range(count) if i != j and matrix[i, j]),
        key=lambda row: (-row["count"], row["true"], row["predicted"]),
    )
    return {
        "count": len(targets), "accuracy": float(correct.sum() / len(targets)),
        "macro_recall": float(recall.mean()), "macro_f1": float(f1.mean()),
        "per_class": per_class, "top_confusions_by_count": confusions[:12],
        "confusion_matrix": matrix.tolist(),
    }
