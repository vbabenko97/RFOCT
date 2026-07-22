"""Classification metrics used by RFOCT's split and OOB objectives."""

# Derived from research code: Copyright © 2020-2023 Vitalii Babenko.
# Historical notice: All rights reserved.

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return zero for an undefined ratio on a valid degenerate subset."""
    return 0.0 if denominator == 0 else float(numerator / denominator)


def get_mcc(tp: float, tn: float, fp: float, fn: float) -> float:
    """Calculate Matthews correlation coefficient, returning zero if undefined."""
    numerator = tp * tn - fp * fn
    denominator = float(np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return _safe_ratio(numerator, denominator)


def get_confusion_matrix(y_true: Sequence[object], y_pred: Sequence[object]) -> dict[str, float]:
    """Return the historical binary confusion matrix for encoded labels 0 and 1."""
    truth = np.asarray(y_true)
    prediction = np.asarray(y_pred)
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return {
        "TP": float(np.sum((truth == prediction) & (truth == 1))),
        "FN": float(np.sum((truth != prediction) & (truth == 1))),
        "FP": float(np.sum((truth != prediction) & (truth == 0))),
        "TN": float(np.sum((truth == prediction) & (truth == 0))),
    }


def calculate_metrics(cm: Mapping[str, float]) -> dict[str, float]:
    """Calculate binary metrics safely on valid single-class subsets."""
    tp, fn, fp, tn = cm["TP"], cm["FN"], cm["FP"], cm["TN"]
    return {
        "accuracy": _safe_ratio(tp + tn, tp + fp + fn + tn),
        "sensitivity": _safe_ratio(tp, tp + fn),
        "specificity": _safe_ratio(tn, tn + fp),
        "precision": _safe_ratio(tp, tp + fp),
        "fscore": _safe_ratio(2 * tp, 2 * tp + fp + fn),
        "mcc": get_mcc(tp, tn, fp, fn),
    }


def calculate_mcc_macro(
    y_true: Sequence[object],
    y_pred: Sequence[object],
    labels: Sequence[object] | None = None,
) -> float:
    """Calculate macro one-vs-rest MCC for multiclass targets."""
    truth = np.asarray(y_true)
    prediction = np.asarray(y_pred)
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    label_values = (
        np.unique(np.concatenate([truth, prediction])) if labels is None else np.asarray(labels)
    )
    if label_values.size == 0:
        return 0.0

    scores = []
    for label in label_values:
        true_positive = truth == label
        predicted_positive = prediction == label
        tp = float(np.sum(true_positive & predicted_positive))
        fp = float(np.sum(~true_positive & predicted_positive))
        fn = float(np.sum(true_positive & ~predicted_positive))
        tn = float(truth.size - tp - fp - fn)
        scores.append(get_mcc(tp, tn, fp, fn))
    return float(np.mean(scores))


__all__ = ["calculate_mcc_macro", "calculate_metrics", "get_confusion_matrix", "get_mcc"]
