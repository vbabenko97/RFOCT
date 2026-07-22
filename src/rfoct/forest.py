"""RFOCT forest construction and weighted voting."""

# Derived from research code: Copyright © 2019-2023 Vitalii Babenko.
# Historical notice: All rights reserved.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import monotonic

import numpy as np

from .genetic_algorithm import genetic_tree
from .optimal_complexity_tree import tree_prediction
from .weighting import ahp_weights, weights_from_powers


def get_weights(priority_list: Sequence[float]) -> list[float]:
    """Return AHP weights for legacy callers passing quality levels directly."""
    return ahp_weights(priority_list).tolist()


def get_forest(
    number_of_trees: int,
    ga_settings: Mapping[str, object],
    k: int,
    n: int,
    data: Mapping[str, np.ndarray],
    test_size: float,
    features: np.ndarray,
    tree_settings: Mapping[str, object],
    voting: str,
    seed_offset: int = 0,
    verbose: bool = True,
) -> list[dict[str, object]]:
    """Build bootstrap/OOB RFOCT trees and assign voting weights."""
    if "X_test" in data or "y_test" in data:
        raise ValueError(
            "Leakage guard: pass only X_train and y_train; evaluate held-out data separately"
        )
    x_train = np.asarray(data["X_train"])
    y_train = np.asarray(data["y_train"])
    classes = np.unique(y_train).tolist()
    forest: list[dict[str, object]] = []
    for tree_number in range(1, number_of_trees + 1):
        if verbose:
            print("Tree #", tree_number)
        started_at = monotonic()
        result = genetic_tree(
            tree_number + seed_offset,
            ga_settings,
            k,
            n,
            x_train,
            y_train,
            test_size,
            features,
            tree_settings,
            verbose=verbose,
        )
        if verbose:
            print("Total time: ", monotonic() - started_at)
        oob = np.asarray(result["best_oob_idx"], dtype=int).tolist()
        forest.append(
            {
                "id": tree_number,
                "tree": result["best_tree"],
                "features": np.asarray(result["best_features"]).tolist(),
                "power": float(result["best_power"]),
                "oob_idx": oob,
                "classes": classes,
            }
        )

    weights = (
        weights_from_powers([float(item["power"]) for item in forest])
        if voting == "weighted"
        else np.full(number_of_trees, 1.0 / number_of_trees)
    )
    for item, weight in zip(forest, weights, strict=True):
        item["weight"] = float(weight)
    return forest


def forest_prediction(
    forest: Sequence[Mapping[str, object]], x: np.ndarray, features: Sequence[str]
) -> dict[str, object]:
    """Predict with weighted class voting across a fitted forest."""
    x = np.asarray(x)
    if not forest:
        return {"y_pred": np.array([]), "y_proba": np.array([])}
    classes = list(forest[0]["classes"])
    is_multiclass = len(classes) > 2
    feature_to_index = {name: index for index, name in enumerate(features)}
    if is_multiclass:
        class_to_index = {label: index for index, label in enumerate(classes)}
        scores = np.zeros((x.shape[0], len(classes)), dtype=float)
    else:
        positive_score = np.zeros(x.shape[0], dtype=float)

    for item in forest:
        tree_features = list(item["features"])
        feature_indexes = [feature_to_index[name] for name in tree_features]
        prediction = tree_prediction(item["tree"], x[:, feature_indexes], tree_features)
        weight = float(item["weight"])
        if is_multiclass:
            prediction_indexes = np.asarray(
                [class_to_index[label] for label in prediction], dtype=int
            )
            np.add.at(scores, (np.arange(x.shape[0]), prediction_indexes), weight)
        else:
            positive_score += prediction.astype(float) * weight

    if is_multiclass:
        class_array = np.asarray(classes)
        return {
            "y_pred": class_array[np.argmax(scores, axis=1)],
            "y_proba": scores,
            "classes": classes,
        }
    return {
        "y_pred": np.where(positive_score < 0.5, 0, 1),
        "y_proba": positive_score,
    }


__all__ = ["forest_prediction", "get_forest", "get_weights"]
