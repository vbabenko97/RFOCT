"""Self-organization forest for binary classification of liver tissue.

Builds a custom Random Forest-like ensemble of threshold-based decision trees
from ultrasound image features. For each sensor type, the algorithm:
1. Splits training data 75/25 into train/test subsets
2. Grows a forest by iteratively finding optimal thresholds per feature
3. Selects the best-performing sub-forest on the exam sample via F-score
4. Reports accuracy, sensitivity, and specificity on exam and validation sets

Master thesis project — Igor Sikorsky Kyiv Polytechnic Institute.
Copyright 2020. All rights reserved.
Authors: Vitalii Babenko
Contacts: vbabenko2191@gmail.com
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POSITIVE_CLASS = 1
NEGATIVE_CLASS = 2

FIRST_BRANCH = 1
SECOND_BRANCH = 2

CLASS_COLUMN = "class"
TRAIN_FILE_SUFFIX = "(train).xlsx"
EXAM_FILE_SUFFIX = "(exam).xlsx"
VALIDATION_FILE_SUFFIX = "(validation).xlsx"

TEST_SIZE = 0.25
RANDOM_STATE = 0
METRIC_CLASS_WEIGHT = 0.5

# WEIGHT_LIST = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
WEIGHT_LIST = [0.9]
SENSOR_LIST = ["convex", "linear", "reinforced", "xmixed", "ymixed"]
FEATURE_LIMIT_LIST = [3, 6, 7, 7, 7]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """Train and test arrays for one tree-expansion subset."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


@dataclass(frozen=True, slots=True)
class ThresholdResult:
    """Best threshold and quality statistics for one feature."""

    threshold: float
    value: float
    side: int
    false_positives: int
    false_negatives: int


@dataclass(frozen=True, slots=True)
class ThresholdCollection:
    """Threshold search results for every feature in a dataset split."""

    thresholds: list[float]
    sides: list[int]
    false_positives: list[int]
    false_negatives: list[int]
    train_values: np.ndarray
    test_values: np.ndarray


@dataclass(frozen=True, slots=True)
class TreeNode:
    """One threshold node inside a tree path."""

    feature: str
    side: int
    threshold: float
    train_value: float
    test_value: float
    false_positives: int
    false_negatives: int
    leaf_number: int
    level_number: int
    previous_leaf: int
    previous_side: int


@dataclass(slots=True)
class ForestTree:
    """Ordered node path that defines one tree."""

    nodes: list[TreeNode]


@dataclass(slots=True)
class PendingBranches:
    """Mutable queues for subsets that still need tree expansion."""

    X_train_list: list[np.ndarray]
    y_train_list: list[np.ndarray]
    X_test_list: list[np.ndarray]
    y_test_list: list[np.ndarray]
    level_number_list: list[int]
    previous_leaf_list: list[int]
    previous_side_list: list[int]
    tree_index_list: list[int]

    def append(
        self,
        data_split: DatasetSplit,
        level_number: int,
        previous_leaf: int,
        previous_side: int,
        tree_index: int,
    ) -> None:
        """Append one pending branch expansion to the shared queues."""
        self.X_train_list.append(data_split.X_train)
        self.y_train_list.append(data_split.y_train)
        self.X_test_list.append(data_split.X_test)
        self.y_test_list.append(data_split.y_test)
        self.level_number_list.append(level_number)
        self.previous_leaf_list.append(previous_leaf)
        self.previous_side_list.append(previous_side)
        self.tree_index_list.append(tree_index)


@dataclass(frozen=True, slots=True)
class NodeExpansionContext:
    """Metadata required when creating new nodes for one split."""

    column_names: list[str]
    test_weight: float
    leaf_number: int
    tree_index: int
    level_number: int
    previous_leaf: int
    previous_side: int
    max_tree_index: int
    max_features: int


# ---------------------------------------------------------------------------
# Threshold helpers
# ---------------------------------------------------------------------------


def get_branch_mask(
    feature_column: np.ndarray,
    threshold: float,
    side: int,
    branch: int,
) -> np.ndarray:
    """Build the boolean mask for one branch of the threshold split.

    Parameters
    ----------
    feature_column : array of feature values
    threshold : split threshold
    side : which side of the split was selected (FIRST_BRANCH or SECOND_BRANCH)
    branch : which branch to select (FIRST_BRANCH or SECOND_BRANCH)

    Returns
    -------
    Boolean mask selecting the rows that belong to *branch*.
    """
    if side == branch:
        return feature_column < threshold
    return feature_column >= threshold


def get_branch_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_index: int,
    threshold: float,
    side: int,
    branch: int,
) -> DatasetSplit:
    """Return train and test subsets that belong to the requested branch.

    Replaces the original ``get_X1_y1`` / ``get_X2_y2`` pair.

    Parameters
    ----------
    X_train, y_train, X_test, y_test : current split arrays
    feature_index : column index of the feature to split on
    threshold : split threshold
    side : which side was selected during threshold search
    branch : FIRST_BRANCH or SECOND_BRANCH

    Returns
    -------
    DatasetSplit with the filtered arrays.
    """
    train_mask = get_branch_mask(X_train[:, feature_index], threshold, side, branch)
    test_mask = get_branch_mask(X_test[:, feature_index], threshold, side, branch)
    return DatasetSplit(
        X_train=X_train[train_mask],
        y_train=y_train[train_mask],
        X_test=X_test[test_mask],
        y_test=y_test[test_mask],
    )


def calculate_value(
    true_positive: int,
    true_negative: int,
    num_of_pos: int,
    num_of_neg: int,
) -> float:
    """Calculate the balanced accuracy from TP / TN counts.

    Parameters
    ----------
    true_positive, true_negative : correct-class counts
    num_of_pos, num_of_neg : total class counts

    Returns
    -------
    Balanced accuracy value (0..1).
    """
    if num_of_pos != 0 and num_of_neg != 0:
        return ((true_positive / num_of_pos) + (true_negative / num_of_neg)) / 2
    if num_of_pos == 0:
        return true_negative / num_of_neg
    return true_positive / num_of_pos


def get_true_counts(
    feature_column: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    side: int,
) -> tuple[int, int]:
    """Count true positives and true negatives for a fixed threshold side.

    Parameters
    ----------
    feature_column : array of feature values
    labels : class labels (POSITIVE_CLASS / NEGATIVE_CLASS)
    threshold : split threshold
    side : FIRST_BRANCH or SECOND_BRANCH

    Returns
    -------
    (true_positive, true_negative) counts.
    """
    if side == FIRST_BRANCH:
        positive_mask = feature_column < threshold
    else:
        positive_mask = feature_column >= threshold
    negative_mask = ~positive_mask

    true_positive = int(np.sum(positive_mask & (labels == POSITIVE_CLASS)))
    true_negative = int(np.sum(negative_mask & (labels == NEGATIVE_CLASS)))
    return true_positive, true_negative


# ---------------------------------------------------------------------------
# Threshold search
# ---------------------------------------------------------------------------


def find_threshold_of_x(
    sorted_values: np.ndarray,
    feature_column: np.ndarray,
    labels: np.ndarray,
    num_of_pos: int,
    num_of_neg: int,
) -> ThresholdResult:
    """Find the best threshold, side, and error counts for one feature.

    Parameters
    ----------
    sorted_values : feature values sorted ascending
    feature_column : unsorted feature values (for counting)
    labels : class labels
    num_of_pos, num_of_neg : total class counts

    Returns
    -------
    ThresholdResult with the best threshold, value, side, FP, FN.
    """
    threshold_list: list[float] = []
    tp_side_one_list: list[int] = []
    tn_side_one_list: list[int] = []
    tp_side_two_list: list[int] = []
    tn_side_two_list: list[int] = []
    value_side_one_list: list[float] = []
    value_side_two_list: list[float] = []

    if sorted_values.shape[0] > 2:
        threshold_indexes = range(1, sorted_values.shape[0] - 1)
    else:
        threshold_indexes = [1]

    single_threshold_case = sorted_values.shape[0] <= 2

    for threshold_index in threshold_indexes:
        threshold = sorted_values[threshold_index]

        tp_side_one, tn_side_one = get_true_counts(
            feature_column, labels, threshold, FIRST_BRANCH,
        )
        tp_side_two, tn_side_two = get_true_counts(
            feature_column, labels, threshold, SECOND_BRANCH,
        )

        threshold_list.append(threshold)
        tp_side_one_list.append(tp_side_one)
        tn_side_one_list.append(tn_side_one)
        tp_side_two_list.append(tp_side_two)
        tn_side_two_list.append(tn_side_two)

        value_side_one = calculate_value(
            tp_side_one, tn_side_one, num_of_pos, num_of_neg,
        )

        if single_threshold_case and num_of_pos > 0 and num_of_neg == 0:
            # BUG PRESERVED: original line 142 uses TP1 instead of TP2.
            # The else branch (num_of_neg == 0) in the single-threshold path
            # computed value_list2 as TP1/num_of_pos rather than TP2/num_of_pos.
            value_side_two = tp_side_one / num_of_pos
        else:
            value_side_two = calculate_value(
                tp_side_two, tn_side_two, num_of_pos, num_of_neg,
            )

        value_side_one_list.append(value_side_one)
        value_side_two_list.append(value_side_two)

    best_side_one_value = max(value_side_one_list)
    best_side_two_value = max(value_side_two_list)

    if best_side_one_value > best_side_two_value:
        best_index = value_side_one_list.index(best_side_one_value)
        true_positive = tp_side_one_list[best_index]
        true_negative = tn_side_one_list[best_index]
        value = best_side_one_value
        side = FIRST_BRANCH
    else:
        best_index = value_side_two_list.index(best_side_two_value)
        true_positive = tp_side_two_list[best_index]
        true_negative = tn_side_two_list[best_index]
        value = best_side_two_value
        side = SECOND_BRANCH

    threshold = threshold_list[best_index]
    false_positives = num_of_pos - true_positive
    false_negatives = num_of_neg - true_negative
    return ThresholdResult(
        threshold=threshold,
        value=value,
        side=side,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def find_thresholds(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> ThresholdCollection:
    """Find the best threshold and value for every feature.

    Parameters
    ----------
    X_train, y_train : training split
    X_test, y_test : test split

    Returns
    -------
    ThresholdCollection with per-feature threshold results and test values.
    """
    num_of_pos = int(np.sum(y_train == POSITIVE_CLASS))
    num_of_neg = int(np.sum(y_train == NEGATIVE_CLASS))
    threshold_list: list[float] = []
    train_value_list: list[float] = []
    side_list: list[int] = []
    false_positive_list: list[int] = []
    false_negative_list: list[int] = []

    for feature_index in range(X_train.shape[1]):
        feature_column = X_train[:, feature_index]
        sorted_values = feature_column.copy()
        sorted_values.sort()

        result = find_threshold_of_x(
            sorted_values, feature_column, y_train, num_of_pos, num_of_neg,
        )
        threshold_list.append(result.threshold)
        train_value_list.append(result.value)
        side_list.append(result.side)
        false_positive_list.append(result.false_positives)
        false_negative_list.append(result.false_negatives)

    train_values = np.asarray(train_value_list)

    # Evaluate thresholds on test split
    num_of_pos = int(np.sum(y_test == POSITIVE_CLASS))
    num_of_neg = int(np.sum(y_test == NEGATIVE_CLASS))

    if y_test.shape[0] > 0:
        test_value_list: list[float] = []
        for feature_index, threshold in enumerate(threshold_list):
            feature_column = X_test[:, feature_index]
            true_positive, true_negative = get_true_counts(
                feature_column, y_test, threshold, side_list[feature_index],
            )
            test_value_list.append(
                calculate_value(true_positive, true_negative, num_of_pos, num_of_neg)
            )
        test_values = np.asarray(test_value_list)
    else:
        test_values = np.ones(X_train.shape[1])

    return ThresholdCollection(
        thresholds=threshold_list,
        sides=side_list,
        false_positives=false_positive_list,
        false_negatives=false_negative_list,
        train_values=train_values,
        test_values=test_values,
    )


# ---------------------------------------------------------------------------
# Feature ranking helpers
# ---------------------------------------------------------------------------


def build_complex_value_frame(
    train_values: np.ndarray,
    test_values: np.ndarray,
    test_weight: float,
) -> pd.DataFrame:
    """Build the ranking DataFrame used to order features by quality.

    Parameters
    ----------
    train_values, test_values : per-feature quality scores
    test_weight : weight given to test performance (0..1)

    Returns
    -------
    DataFrame sorted by complex_value descending.
    """
    complex_values = (1 - test_weight) * train_values + test_weight * test_values
    frame = pd.DataFrame({
        "train_value": train_values,
        "test_value": test_values,
        "complex_value": complex_values,
    })
    return frame.sort_values(
        ["complex_value", "test_value", "train_value"],
        ascending=[False, False, False],
    )


def get_value_on_next_level(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_weight: float,
) -> float:
    """Calculate the best complex feature value for the next tree level.

    Parameters
    ----------
    X_train, y_train, X_test, y_test : current split
    test_weight : weight for test performance

    Returns
    -------
    Best complex value (float).
    """
    thresholds = find_thresholds(X_train, y_train, X_test, y_test)
    frame = build_complex_value_frame(
        thresholds.train_values, thresholds.test_values, test_weight,
    )
    return float(frame["complex_value"].values[0])


def build_next_level_value_list(
    data_split: DatasetSplit,
    thresholds: ThresholdCollection,
    test_weight: float,
    num_features: int,
) -> list[float]:
    """Evaluate the average next-level value for every candidate feature.

    Parameters
    ----------
    data_split : current train/test arrays
    thresholds : per-feature threshold results
    test_weight : weight for test performance
    num_features : number of features to evaluate

    Returns
    -------
    List of average values, one per feature.
    """
    value_list: list[float] = []
    for feature_index in range(num_features):
        if thresholds.false_positives[feature_index] > 0:
            first_split = get_branch_split(
                data_split.X_train, data_split.y_train,
                data_split.X_test, data_split.y_test,
                feature_index,
                thresholds.thresholds[feature_index],
                thresholds.sides[feature_index],
                FIRST_BRANCH,
            )
            if first_split.y_train.shape[0] > 1:
                first_value = get_value_on_next_level(
                    first_split.X_train, first_split.y_train,
                    first_split.X_test, first_split.y_test,
                    test_weight,
                )
            else:
                first_value = 0.0
        else:
            first_value = 1.0

        if thresholds.false_negatives[feature_index] > 0:
            second_split = get_branch_split(
                data_split.X_train, data_split.y_train,
                data_split.X_test, data_split.y_test,
                feature_index,
                thresholds.thresholds[feature_index],
                thresholds.sides[feature_index],
                SECOND_BRANCH,
            )
            if second_split.y_train.shape[0] > 1:
                second_value = get_value_on_next_level(
                    second_split.X_train, second_split.y_train,
                    second_split.X_test, second_split.y_test,
                    test_weight,
                )
            else:
                second_value = 0.0
        else:
            second_value = 1.0

        value_list.append((first_value + second_value) / 2)
    return value_list


# ---------------------------------------------------------------------------
# Node and forest construction
# ---------------------------------------------------------------------------


def get_new_nodes(
    data_split: DatasetSplit,
    context: NodeExpansionContext,
    pending: PendingBranches,
) -> list[TreeNode]:
    """Create the next set of tree nodes and queue their child branches.

    Parameters
    ----------
    data_split : current train/test arrays for this node
    context : metadata (feature names, weights, leaf/level tracking)
    pending : mutable queues for branches that still need expansion

    Returns
    -------
    List of TreeNode objects (one per selected feature, up to max_features).
    """
    thresholds = find_thresholds(
        data_split.X_train, data_split.y_train,
        data_split.X_test, data_split.y_test,
    )

    if max(thresholds.train_values) < 1.0:
        value_list = build_next_level_value_list(
            data_split, thresholds, context.test_weight, len(context.column_names),
        )
        ranking_frame = pd.DataFrame({"value": value_list}).sort_values(
            ["value"], ascending=[False],
        )
    else:
        ranking_frame = build_complex_value_frame(
            thresholds.train_values, thresholds.test_values, context.test_weight,
        )

    index_list = ranking_frame.index.tolist()[: context.max_features]
    node_list: list[TreeNode] = []
    max_tree_index = context.max_tree_index
    temp_index = 0

    for feature_index in index_list:
        side = thresholds.sides[feature_index]
        threshold = thresholds.thresholds[feature_index]
        false_positives = thresholds.false_positives[feature_index]
        false_negatives = thresholds.false_negatives[feature_index]

        if false_positives > 0:
            first_split = get_branch_split(
                data_split.X_train, data_split.y_train,
                data_split.X_test, data_split.y_test,
                feature_index, threshold, side, FIRST_BRANCH,
            )
            if first_split.y_train.shape[0] > 1:
                pending.append(
                    data_split=first_split,
                    level_number=context.level_number,
                    previous_leaf=context.leaf_number,
                    previous_side=FIRST_BRANCH,
                    tree_index=(
                        context.tree_index if temp_index == 0 else max_tree_index
                    ),
                )

        if false_negatives > 0:
            second_split = get_branch_split(
                data_split.X_train, data_split.y_train,
                data_split.X_test, data_split.y_test,
                feature_index, threshold, side, SECOND_BRANCH,
            )
            if second_split.y_train.shape[0] > 1:
                pending.append(
                    data_split=second_split,
                    level_number=context.level_number,
                    previous_leaf=context.leaf_number,
                    previous_side=SECOND_BRANCH,
                    tree_index=(
                        context.tree_index if temp_index == 0 else max_tree_index
                    ),
                )

        node_list.append(TreeNode(
            feature=context.column_names[feature_index],
            side=side,
            threshold=threshold,
            train_value=float(thresholds.train_values[feature_index]),
            test_value=float(thresholds.test_values[feature_index]),
            false_positives=false_positives,
            false_negatives=false_negatives,
            leaf_number=context.leaf_number,
            level_number=context.level_number,
            previous_leaf=context.previous_leaf,
            previous_side=context.previous_side,
        ))
        max_tree_index += 1
        temp_index += 1

    return node_list


def get_forest(
    test_weight: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    column_names: list[str],
    max_features: int,
) -> list[ForestTree]:
    """Grow the forest by repeatedly expanding queued branches.

    Parameters
    ----------
    test_weight : weight for test performance in feature ranking
    X_train, y_train, X_test, y_test : train/test split
    column_names : feature names from the training DataFrame
    max_features : maximum features (F) selected per level

    Returns
    -------
    List of ForestTree objects.
    """
    tree_list: list[ForestTree] = []
    leaf_number = 1
    level_number = 1
    tree_index = 0

    pending = PendingBranches(
        X_train_list=[], y_train_list=[],
        X_test_list=[], y_test_list=[],
        level_number_list=[], previous_leaf_list=[],
        previous_side_list=[], tree_index_list=[],
    )

    root_split = DatasetSplit(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
    )
    root_nodes = get_new_nodes(
        data_split=root_split,
        context=NodeExpansionContext(
            column_names=column_names,
            test_weight=test_weight,
            leaf_number=leaf_number,
            tree_index=tree_index,
            level_number=level_number,
            previous_leaf=0,
            previous_side=0,
            max_tree_index=0,
            max_features=max_features,
        ),
        pending=pending,
    )
    for node in root_nodes:
        tree_list.append(ForestTree(nodes=[node]))

    count_list = np.zeros(max_features)
    queue_index = 0

    while queue_index < len(pending.previous_leaf_list):
        branch_split = DatasetSplit(
            X_train=pending.X_train_list[queue_index],
            y_train=pending.y_train_list[queue_index],
            X_test=pending.X_test_list[queue_index],
            y_test=pending.y_test_list[queue_index],
        )
        leaf_number += 1
        level_number = pending.level_number_list[queue_index] + 1
        previous_leaf = pending.previous_leaf_list[queue_index]
        previous_side = pending.previous_side_list[queue_index]
        tree_index = pending.tree_index_list[queue_index]
        max_tree_index = max(pending.tree_index_list)

        new_nodes = get_new_nodes(
            data_split=branch_split,
            context=NodeExpansionContext(
                column_names=column_names,
                test_weight=test_weight,
                leaf_number=leaf_number,
                tree_index=tree_index,
                level_number=level_number,
                previous_leaf=previous_leaf,
                previous_side=previous_side,
                max_tree_index=max_tree_index,
                max_features=max_features,
            ),
            pending=pending,
        )

        temp_tree = copy.deepcopy(tree_list[tree_index].nodes)
        temp_index = 0
        for new_node in new_nodes:
            tree_nodes = copy.deepcopy(temp_tree)
            if np.sum(count_list) < max_features ** 2:
                if count_list[tree_index] < max_features:
                    count_list[tree_index] += 1
            tree_nodes.append(new_node)
            if temp_index == 0:
                tree_list[tree_index] = ForestTree(nodes=tree_nodes)
            else:
                tree_list.append(ForestTree(nodes=tree_nodes))
            temp_index += 1

        queue_index += 1

    return tree_list


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def get_tree_prediction(tree: ForestTree, exam_object: pd.Series) -> int:
    """Traverse one tree and return its predicted class for an exam object.

    Parameters
    ----------
    tree : a single ForestTree
    exam_object : one row from the exam DataFrame

    Returns
    -------
    Predicted class (POSITIVE_CLASS or NEGATIVE_CLASS).
    """
    level = 1
    node_index = 0
    predicted_class = 0

    while True:
        node = tree.nodes[node_index]
        if node.side == FIRST_BRANCH:
            if exam_object[node.feature] < node.threshold:
                predicted_class = POSITIVE_CLASS
            else:
                predicted_class = NEGATIVE_CLASS
        else:
            if exam_object[node.feature] < node.threshold:
                predicted_class = NEGATIVE_CLASS
            else:
                predicted_class = POSITIVE_CLASS

        # Find next node whose previous_leaf matches current level
        # and previous_side matches the predicted class
        next_node_index = None
        for candidate_index, candidate in enumerate(tree.nodes):
            if (
                candidate.previous_leaf == level
                and candidate.previous_side == predicted_class
            ):
                next_node_index = candidate_index
                break

        if next_node_index is None:
            break
        node_index = next_node_index
        level = tree.nodes[node_index].leaf_number

    return predicted_class


def get_exam_value(
    tree_list: list[ForestTree],
    exam_data: pd.DataFrame,
    y_exam: np.ndarray,
) -> tuple[float, float, float]:
    """Calculate accuracy, sensitivity, and specificity for a forest.

    Parameters
    ----------
    tree_list : list of ForestTree objects
    exam_data : exam DataFrame (features + class column)
    y_exam : exam class labels

    Returns
    -------
    (accuracy, sensitivity, specificity) tuple.
    """
    num_of_pos = int(np.sum(y_exam == POSITIVE_CLASS))
    num_of_neg = int(np.sum(y_exam == NEGATIVE_CLASS))
    correct_count = 0
    true_positive = 0
    true_negative = 0

    for object_index in range(exam_data.shape[0]):
        exam_object = exam_data.loc[object_index]
        predictions = [get_tree_prediction(tree, exam_object) for tree in tree_list]
        prediction_array = np.asarray(predictions)

        if np.sum(prediction_array == POSITIVE_CLASS) > np.sum(
            prediction_array == NEGATIVE_CLASS
        ):
            predicted_class = POSITIVE_CLASS
        else:
            predicted_class = NEGATIVE_CLASS

        if y_exam[object_index] == predicted_class:
            correct_count += 1
            if y_exam[object_index] == POSITIVE_CLASS:
                true_positive += 1
            else:
                true_negative += 1

    return (
        correct_count / y_exam.shape[0],
        true_positive / num_of_pos,
        true_negative / num_of_neg,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the sensor-by-sensor training and evaluation workflow."""
    for sensor_type, max_features in zip(SENSOR_LIST, FEATURE_LIMIT_LIST):
        print("Sensor type: ", sensor_type)

        train_data = pd.read_excel(sensor_type + TRAIN_FILE_SUFFIX)
        exam_data = pd.read_excel(sensor_type + EXAM_FILE_SUFFIX)
        validation_data = pd.read_excel(sensor_type + VALIDATION_FILE_SUFFIX)

        column_names = list(train_data.columns[:-1])

        labels = train_data[CLASS_COLUMN].values
        features = train_data.drop([CLASS_COLUMN], axis=1).values
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        )

        y_exam = exam_data[CLASS_COLUMN].values
        y_validation = validation_data[CLASS_COLUMN].values

        best_value = 0
        for test_weight in WEIGHT_LIST:
            forest = get_forest(
                test_weight=test_weight,
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                column_names=column_names,
                max_features=max_features,
            )

            leaf_list = [
                max(node.leaf_number for node in tree.nodes) for tree in forest
            ]
            level_list = [
                max(node.level_number for node in tree.nodes) for tree in forest
            ]
            criterion = pd.DataFrame({
                "number_of_leafs": leaf_list,
                "number_of_levels": level_list,
            })
            criterion = criterion.sort_values(["number_of_leafs", "number_of_levels"])

            for forest_size in range(1, 21):
                best_tree_indexes = criterion.index.tolist()[:forest_size]
                sub_forest = [forest[idx] for idx in best_tree_indexes]
                accuracy, sensitivity, specificity = get_exam_value(
                    sub_forest, exam_data, y_exam,
                )
                f_value = (
                    METRIC_CLASS_WEIGHT * sensitivity
                    + METRIC_CLASS_WEIGHT * specificity
                )
                if f_value > best_value:
                    best_value = f_value
                    best_forest = copy.deepcopy(sub_forest)
                    best_weight = test_weight
                    optimal_t = forest_size
                    top_accuracy = accuracy
                    top_sensitivity = sensitivity
                    top_specificity = specificity

        print("Exam result:")
        print(" - Best weight: ", best_weight)
        print(" - Optimal t: ", optimal_t)
        print(" - Top accuracy: ", top_accuracy)
        print(" - Top sensitivity: ", top_sensitivity)
        print(" - Top specificty: ", top_specificity)
        accuracy, sensitivity, specificity = get_exam_value(
            best_forest, validation_data, y_validation,
        )
        print("Validation result:")
        print(" - Accuracy: ", accuracy)
        print(" - Sensitivity: ", sensitivity)
        print(" - Specificity: ", specificity)
        print()


if __name__ == "__main__":
    main()
