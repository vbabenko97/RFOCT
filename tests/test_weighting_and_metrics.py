from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from rfoct.classification_evaluation import (
    calculate_mcc_macro,
    calculate_metrics,
    get_confusion_matrix,
    get_mcc,
)
from rfoct.weighting import ahp_weights, rank_quality_levels, weights_from_powers


def test_ahp_weights_follow_larger_quality_is_better_semantics() -> None:
    assert_allclose(ahp_weights([1, 2, 3]), [1 / 6, 2 / 6, 3 / 6])


def test_power_ranking_preserves_input_order_and_ties() -> None:
    powers = [0.8, 0.2, 0.8, 0.5]

    assert_array_equal(rank_quality_levels(powers), [3, 1, 3, 2])
    assert_allclose(weights_from_powers(powers), [3 / 9, 1 / 9, 3 / 9, 2 / 9])


def test_equal_powers_receive_equal_normalized_weights() -> None:
    assert_allclose(weights_from_powers([0.4, 0.4, 0.4]), [1 / 3, 1 / 3, 1 / 3])


@pytest.mark.parametrize(
    ("y_true", "expected"),
    [
        (
            np.zeros(4, dtype=int),
            {
                "accuracy": 1.0,
                "sensitivity": 0.0,
                "specificity": 1.0,
                "precision": 0.0,
                "fscore": 0.0,
                "mcc": 0.0,
            },
        ),
        (
            np.ones(4, dtype=int),
            {
                "accuracy": 1.0,
                "sensitivity": 1.0,
                "specificity": 0.0,
                "precision": 1.0,
                "fscore": 1.0,
                "mcc": 0.0,
            },
        ),
    ],
)
def test_metrics_are_zero_safe_on_single_class_subsets(
    y_true: np.ndarray,
    expected: dict[str, float],
) -> None:
    metrics = calculate_metrics(get_confusion_matrix(y_true, y_true.copy()))

    assert metrics == expected
    assert calculate_mcc_macro(y_true, y_true.copy()) == 0.0


def test_metrics_are_zero_safe_on_empty_subset() -> None:
    metrics = calculate_metrics({"TP": 0.0, "TN": 0.0, "FP": 0.0, "FN": 0.0})

    assert set(metrics.values()) == {0.0}
    assert get_mcc(0.0, 0.0, 0.0, 0.0) == 0.0
