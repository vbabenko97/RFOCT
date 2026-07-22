"""Optimal-complexity tree construction and prediction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def calculate_mcc_vectorized(
    tp: np.ndarray, tn: np.ndarray, fp: np.ndarray, fn: np.ndarray
) -> np.ndarray:
    """Calculate MCC element-wise, returning zero where it is undefined."""
    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )


def _gini_from_counts(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0
    probabilities = counts.astype(float) / total
    return float(1.0 - np.sum(probabilities * probabilities))


def find_best_split_vectorized(x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    """Find the binary threshold and orientation with maximum MCC."""
    if x.size == 0 or np.min(x) == np.max(x):
        return {"threshold": 0.0, "side": 0, "FN": -1, "FP": -1, "value": -1.0}

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    valid = np.flatnonzero(np.diff(x_sorted) != 0)
    if valid.size == 0:
        return {"threshold": 0.0, "side": 0, "FN": -1, "FP": -1, "value": -1.0}

    total_positive = int(np.sum(y_sorted))
    left_positive = np.cumsum(y_sorted)
    left_count = np.arange(1, y_sorted.size + 1)
    left_negative = left_count - left_positive
    right_positive = total_positive - left_positive
    right_negative = y_sorted.size - total_positive - left_negative

    mcc_side_1 = calculate_mcc_vectorized(
        right_positive[valid],
        left_negative[valid],
        right_negative[valid],
        left_positive[valid],
    )
    mcc_side_2 = calculate_mcc_vectorized(
        left_positive[valid],
        right_negative[valid],
        left_negative[valid],
        right_positive[valid],
    )
    best_1 = int(np.argmax(mcc_side_1))
    best_2 = int(np.argmax(mcc_side_2))
    if mcc_side_1[best_1] > mcc_side_2[best_2]:
        split_index = int(valid[best_1])
        side = 1
        value = float(mcc_side_1[best_1])
        false_negative = int(left_positive[split_index])
        false_positive = int(right_negative[split_index])
    else:
        split_index = int(valid[best_2])
        side = 2
        value = float(mcc_side_2[best_2])
        false_negative = int(right_positive[split_index])
        false_positive = int(left_negative[split_index])
    threshold = float((x_sorted[split_index] + x_sorted[split_index + 1]) / 2.0)
    return {
        "threshold": threshold,
        "side": side,
        "FN": false_negative,
        "FP": false_positive,
        "value": value,
    }


def find_best_split_multiclass(
    x: np.ndarray, y: np.ndarray, labels: np.ndarray
) -> dict[str, object]:
    """Find a threshold maximizing macro one-vs-rest MCC."""
    if x.size == 0 or np.min(x) == np.max(x):
        majority = labels[0] if y.size == 0 else labels[np.argmax([np.sum(y == v) for v in labels])]
        return {
            "threshold": 0.0,
            "left_class": majority,
            "right_class": majority,
            "value": -1.0,
        }

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    valid = np.flatnonzero(np.diff(x_sorted) != 0)
    if valid.size == 0:
        majority = labels[np.argmax([np.sum(y == value) for value in labels])]
        return {
            "threshold": 0.0,
            "left_class": majority,
            "right_class": majority,
            "value": -1.0,
        }

    label_to_index = {label: index for index, label in enumerate(labels)}
    encoded = np.array([label_to_index[value] for value in y_sorted], dtype=int)
    one_hot = np.eye(labels.size, dtype=int)[encoded]
    cumulative = np.cumsum(one_hot, axis=0)
    left_counts = cumulative[valid]
    totals = cumulative[-1]
    right_counts = totals - left_counts
    left_sizes = valid + 1
    right_sizes = y_sorted.size - left_sizes
    left_majority = np.argmax(left_counts, axis=1)
    right_majority = np.argmax(right_counts, axis=1)
    left_majority_hot = np.eye(labels.size, dtype=int)[left_majority]
    right_majority_hot = np.eye(labels.size, dtype=int)[right_majority]
    tp = left_majority_hot * left_counts + right_majority_hot * right_counts
    fp = left_majority_hot * (left_sizes[:, None] - left_counts) + right_majority_hot * (
        right_sizes[:, None] - right_counts
    )
    fn = totals - tp
    tn = y_sorted.size - tp - fp - fn
    score = np.mean(calculate_mcc_vectorized(tp, tn, fp, fn), axis=1)
    best = int(np.argmax(score))
    split_index = int(valid[best])
    return {
        "threshold": float((x_sorted[split_index] + x_sorted[split_index + 1]) / 2.0),
        "left_class": labels[left_majority[best]],
        "right_class": labels[right_majority[best]],
        "value": float(score[best]),
    }


def find_best_split_gini(x: np.ndarray, y: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    """Find a threshold maximizing Gini gain."""
    if x.size == 0 or np.min(x) == np.max(x):
        majority = labels[0] if y.size == 0 else labels[np.argmax([np.sum(y == v) for v in labels])]
        return {
            "threshold": 0.0,
            "left_class": majority,
            "right_class": majority,
            "value": -1.0,
        }

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    valid = np.flatnonzero(np.diff(x_sorted) != 0)
    label_to_index = {label: index for index, label in enumerate(labels)}
    encoded = np.array([label_to_index[value] for value in y_sorted], dtype=int)
    cumulative = np.cumsum(np.eye(labels.size, dtype=int)[encoded], axis=0)
    left_counts = cumulative[valid]
    totals = cumulative[-1]
    right_counts = totals - left_counts
    left_sizes = valid + 1
    right_sizes = y_sorted.size - left_sizes
    sample_count = float(y_sorted.size)
    parent_gini = _gini_from_counts(totals)
    left_probabilities = left_counts / left_sizes[:, None]
    right_probabilities = right_counts / right_sizes[:, None]
    left_gini = 1.0 - np.sum(left_probabilities**2, axis=1)
    right_gini = 1.0 - np.sum(right_probabilities**2, axis=1)
    gain = (
        parent_gini
        - (left_sizes / sample_count) * left_gini
        - (right_sizes / sample_count) * right_gini
    )
    best = int(np.argmax(gain))
    split_index = int(valid[best])
    return {
        "threshold": float((x_sorted[split_index] + x_sorted[split_index + 1]) / 2.0),
        "left_class": labels[int(np.argmax(left_counts[best]))],
        "right_class": labels[int(np.argmax(right_counts[best]))],
        "value": float(gain[best]),
    }


def _evaluate_multiclass_thresholds(
    x: np.ndarray,
    y: np.ndarray,
    thresholds: Sequence[float],
    left_classes: Sequence[object],
    right_classes: Sequence[object],
    labels: np.ndarray,
) -> np.ndarray:
    if y.size == 0:
        return np.full(x.shape[1], -1.0)
    scores = np.empty(x.shape[1], dtype=float)
    for index in range(x.shape[1]):
        prediction = np.where(
            x[:, index] < thresholds[index], left_classes[index], right_classes[index]
        )
        per_class = []
        for label in labels:
            true_positive = y == label
            predicted_positive = prediction == label
            tp = np.sum(true_positive & predicted_positive)
            fp = np.sum(~true_positive & predicted_positive)
            fn = np.sum(true_positive & ~predicted_positive)
            tn = y.size - tp - fp - fn
            per_class.append(
                calculate_mcc_vectorized(
                    np.asarray(tp), np.asarray(tn), np.asarray(fp), np.asarray(fn)
                )
            )
        scores[index] = float(np.mean(per_class))
    return scores


def _evaluate_gini_thresholds(
    x: np.ndarray, y: np.ndarray, thresholds: Sequence[float], labels: np.ndarray
) -> np.ndarray:
    if y.size == 0:
        return np.full(x.shape[1], -1.0)
    parent = np.array([np.sum(y == label) for label in labels])
    parent_gini = _gini_from_counts(parent)
    gains = np.empty(x.shape[1], dtype=float)
    for index in range(x.shape[1]):
        left = y[x[:, index] < thresholds[index]]
        right = y[x[:, index] >= thresholds[index]]
        left_counts = np.array([np.sum(left == label) for label in labels])
        right_counts = np.array([np.sum(right == label) for label in labels])
        gains[index] = (
            parent_gini
            - (left.size / y.size) * _gini_from_counts(left_counts)
            - (right.size / y.size) * _gini_from_counts(right_counts)
        )
    return gains


def find_thresholds(
    train_indexes: Sequence[int],
    test_indexes: Sequence[int],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    split_criterion: str = "mcc",
    classes: Sequence[object] | None = None,
) -> dict[str, object]:
    """Evaluate each feature's split on node-local A/B subsets."""
    x_subset = x_train[np.asarray(train_indexes, dtype=int)]
    y_subset = y_train[np.asarray(train_indexes, dtype=int)]
    if classes is None:
        test_labels = y_test[np.asarray(test_indexes, dtype=int)] if test_indexes else []
        labels = np.unique(np.concatenate([y_subset, test_labels]))
    else:
        labels = np.asarray(classes)
    use_gini = split_criterion.lower() == "gini"
    use_class_labels = labels.size > 2 or use_gini

    thresholds: list[float] = []
    sides: list[int] = []
    false_negatives: list[int] = []
    false_positives: list[int] = []
    train_values: list[float] = []
    left_classes: list[object] = []
    right_classes: list[object] = []
    for index in range(x_train.shape[1]):
        if use_class_labels:
            finder = find_best_split_gini if use_gini else find_best_split_multiclass
            result = finder(x_subset[:, index], y_subset, labels)
            sides.append(1)
            false_negatives.append(-1)
            false_positives.append(-1)
            left_classes.append(result["left_class"])
            right_classes.append(result["right_class"])
        else:
            result = find_best_split_vectorized(x_subset[:, index], (y_subset == 1).astype(np.int8))
            sides.append(int(result["side"]))
            false_negatives.append(int(result["FN"]))
            false_positives.append(int(result["FP"]))
        thresholds.append(float(result["threshold"]))
        train_values.append(float(result["value"]))

    if test_indexes:
        test_index_array = np.asarray(test_indexes, dtype=int)
        x_test_subset = x_test[test_index_array]
        y_test_subset = y_test[test_index_array]
        if use_gini:
            test_values = _evaluate_gini_thresholds(
                x_test_subset, y_test_subset, thresholds, labels
            )
        elif use_class_labels:
            test_values = _evaluate_multiclass_thresholds(
                x_test_subset,
                y_test_subset,
                thresholds,
                left_classes,
                right_classes,
                labels,
            )
        else:
            y_positive = (y_test_subset == 1)[:, None]
            comparisons = x_test_subset >= np.asarray(thresholds)[None, :]
            predicted_positive = np.where(
                np.asarray(sides)[None, :] == 1, comparisons, ~comparisons
            )
            tp = np.sum(predicted_positive & y_positive, axis=0)
            fp = np.sum(predicted_positive & ~y_positive, axis=0)
            tn = np.sum(~predicted_positive & ~y_positive, axis=0)
            fn = np.sum(~predicted_positive & y_positive, axis=0)
            test_values = np.where(
                np.asarray(sides) > 0,
                calculate_mcc_vectorized(tp, tn, fp, fn),
                -1.0,
            )
    else:
        test_values = np.ones(x_train.shape[1])

    output: dict[str, object] = {
        "threshold": thresholds,
        "side": sides,
        "FN": false_negatives,
        "FP": false_positives,
        "train_value": np.asarray(train_values),
        "test_value": np.asarray(test_values),
    }
    if use_class_labels:
        output.update(
            left_class=left_classes,
            right_class=right_classes,
            labels=labels.tolist(),
        )
    return output


def _subset_indexes(
    indexes: Sequence[int], feature: np.ndarray, threshold: float, direction: int
) -> list[int]:
    index_array = np.asarray(indexes, dtype=int)
    mask = feature[index_array] < threshold if direction == 1 else feature[index_array] >= threshold
    return index_array[mask].tolist()


def get_new_node(parameters: Mapping[str, object]) -> dict[str, object]:
    """Build one node and identify the impure branches needing child nodes."""
    train_indexes = list(parameters["train_indexes"])
    test_indexes = list(parameters["test_indexes"])
    x_train = np.asarray(parameters["X_train"])
    y_train = np.asarray(parameters["y_train"])
    x_test = np.asarray(parameters["X_test"])
    y_test = np.asarray(parameters["y_test"])
    features = np.asarray(parameters["features"])
    result = find_thresholds(
        train_indexes,
        test_indexes,
        x_train,
        y_train,
        x_test,
        y_test,
        str(parameters.get("split_criterion", "mcc")),
        parameters.get("classes"),
    )
    train_values = np.asarray(result["train_value"])
    test_values = np.asarray(result["test_value"])
    selected = int(np.lexsort((-train_values, -test_values))[0])
    threshold = float(result["threshold"][selected])
    side = int(result["side"][selected])
    feature_column_train = x_train[:, selected]
    feature_column_test = x_test[:, selected]
    is_class_labeled = "left_class" in result

    child_train: list[list[int]] = []
    child_test: list[list[int]] = []
    child_parent: list[int] = []
    child_level: list[int] = []
    child_direction: list[int] = []

    def append_child(actual_direction: int, tree_direction: int) -> None:
        new_train = _subset_indexes(
            train_indexes, feature_column_train, threshold, actual_direction
        )
        if len(new_train) <= 1 or new_train == train_indexes:
            return
        if is_class_labeled and np.unique(y_train[new_train]).size <= 1:
            return
        new_test = _subset_indexes(test_indexes, feature_column_test, threshold, actual_direction)
        child_train.append(new_train)
        child_test.append(new_test)
        child_parent.append(int(parameters["leaf_number"]))
        child_level.append(int(parameters["level_number"]))
        child_direction.append(tree_direction)

    if is_class_labeled:
        append_child(1, 1)
        append_child(2, 2)
    else:
        if result["FN"][selected] > 0:
            append_child(1 if side == 1 else 2, 1)
        if result["FP"][selected] > 0:
            append_child(2 if side == 1 else 1, 2)

    node: dict[str, object] = {
        "feature": features[selected],
        "side": side,
        "threshold": threshold,
        "leaf_number": int(parameters["leaf_number"]),
        "level_number": int(parameters["level_number"]),
        "previous_leaf": int(parameters["previous_leaf"]),
        "previous_direction": int(parameters["previous_direction"]),
    }
    if is_class_labeled:
        node["left_class"] = result["left_class"][selected]
        node["right_class"] = result["right_class"][selected]
    return {
        "node": node,
        "train_list": child_train,
        "test_list": child_test,
        "pll": child_parent,
        "lnl": child_level,
        "pdl": child_direction,
    }


def get_tree(tree_parameters: Mapping[str, object]) -> list[dict[str, object]]:
    """Construct an optimal-complexity tree using A/B split validation."""
    x_train = np.asarray(tree_parameters["X_train"])
    y_train = np.asarray(tree_parameters["y_train"])
    x_test = np.asarray(tree_parameters["X_test"])
    y_test = np.asarray(tree_parameters["y_test"])
    classes = tree_parameters.get("classes")
    if classes is None:
        classes = np.unique(np.concatenate([y_train, y_test]))
    base = {
        "X_train": x_train,
        "y_train": y_train,
        "X_test": x_test,
        "y_test": y_test,
        "features": np.asarray(tree_parameters["features"]),
        "split_criterion": str(tree_parameters.get("split_criterion", "mcc")),
        "classes": np.asarray(classes),
    }
    initial = {
        **base,
        "train_indexes": list(range(y_train.size)),
        "test_indexes": list(range(y_test.size)),
        "level_number": 1,
        "leaf_number": 1,
        "previous_leaf": 0,
        "previous_direction": 0,
    }
    result = get_new_node(initial)
    nodes = [result["node"]]
    train_queue = result["train_list"]
    test_queue = result["test_list"]
    parent_queue = result["pll"]
    level_queue = result["lnl"]
    direction_queue = result["pdl"]
    max_level = tree_parameters["max_level"]
    cursor = 0
    while cursor < len(parent_queue):
        level = level_queue[cursor] + 1
        if isinstance(max_level, int) and level > max_level:
            break
        child = get_new_node(
            {
                **base,
                "train_indexes": train_queue[cursor],
                "test_indexes": test_queue[cursor],
                "level_number": level,
                "leaf_number": len(nodes) + 1,
                "previous_leaf": parent_queue[cursor],
                "previous_direction": direction_queue[cursor],
            }
        )
        nodes.append(child["node"])
        train_queue.extend(child["train_list"])
        test_queue.extend(child["test_list"])
        parent_queue.extend(child["pll"])
        level_queue.extend(child["lnl"])
        direction_queue.extend(child["pdl"])
        cursor += 1
    return nodes


def tree_prediction(
    tree: Sequence[Mapping[str, object]], x: np.ndarray, features: Sequence[str]
) -> np.ndarray:
    """Predict encoded labels for one RFOCT tree."""
    x = np.asarray(x)
    if x.shape[0] == 0:
        return np.array([], dtype=int)
    if not tree:
        return np.zeros(x.shape[0], dtype=int)
    feature_to_index = {name: index for index, name in enumerate(features)}
    children = {
        (int(node["previous_leaf"]), int(node["previous_direction"])): index
        for index, node in enumerate(tree)
        if (int(node["previous_leaf"]), int(node["previous_direction"])) != (0, 0)
    }
    output = np.empty(x.shape[0], dtype=object)
    stack = [(0, np.arange(x.shape[0]))]
    while stack:
        node_index, sample_indexes = stack.pop()
        if sample_indexes.size == 0:
            continue
        node = tree[node_index]
        column = x[sample_indexes, feature_to_index[node["feature"]]]
        class_labeled = "left_class" in node
        if class_labeled:
            left_mask = column < float(node["threshold"])
        else:
            left_mask = (
                column < float(node["threshold"])
                if int(node["side"]) == 1
                else column >= float(node["threshold"])
            )
        parent = int(node["leaf_number"])
        for direction, indexes, terminal_key, binary_label in (
            (1, sample_indexes[left_mask], "left_class", 0),
            (2, sample_indexes[~left_mask], "right_class", 1),
        ):
            child = children.get((parent, direction))
            if child is None:
                output[indexes] = node[terminal_key] if class_labeled else binary_label
            else:
                stack.append((child, indexes))
    try:
        return output.astype(int)
    except (TypeError, ValueError):
        return output


__all__ = [
    "calculate_mcc_vectorized",
    "find_best_split_gini",
    "find_best_split_multiclass",
    "find_best_split_vectorized",
    "find_thresholds",
    "get_new_node",
    "get_tree",
    "tree_prediction",
]
