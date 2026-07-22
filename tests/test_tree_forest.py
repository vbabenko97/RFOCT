from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from rfoct import RFOCTClassifier
from rfoct.forest import get_forest
from rfoct.genetic_algorithm import _safe_stratified_split
from rfoct.optimal_complexity_tree import get_tree, tree_prediction


def test_fitted_forest_preserves_tree_and_oob_invariants(
    binary_model: RFOCTClassifier,
    binary_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, _ = binary_data
    forest = binary_model.forest_

    assert len(forest) == binary_model.n_estimators
    assert sum(tree_object["weight"] for tree_object in forest) == pytest.approx(1.0)

    for expected_id, tree_object in enumerate(forest, start=1):
        assert tree_object["id"] == expected_id
        assert tree_object["tree"]
        assert set(tree_object["features"]) <= set(binary_model.rfoct_feature_names_)

        oob_indexes = tree_object["oob_idx"]
        assert len(oob_indexes) == len(set(oob_indexes))
        assert all(0 <= index < X.shape[0] for index in oob_indexes)

        nodes = tree_object["tree"]
        nodes_by_leaf = {node["leaf_number"]: node for node in nodes}
        assert len(nodes_by_leaf) == len(nodes)
        root = nodes_by_leaf[1]
        assert root["level_number"] == 1
        assert (root["previous_leaf"], root["previous_direction"]) == (0, 0)

        child_links = set()
        for node in nodes:
            assert node["feature"] in tree_object["features"]
            assert node["level_number"] <= binary_model.max_level
            if node is root:
                continue
            parent = nodes_by_leaf[node["previous_leaf"]]
            assert node["previous_direction"] in {1, 2}
            assert node["level_number"] == parent["level_number"] + 1
            link = (node["previous_leaf"], node["previous_direction"])
            assert link not in child_links
            child_links.add(link)


def test_multiclass_forest_nodes_carry_terminal_class_metadata(
    multiclass_model: RFOCTClassifier,
) -> None:
    for tree_object in multiclass_model.forest_:
        assert tree_object["classes"] == [0, 1, 2]
        for node in tree_object["tree"]:
            assert "left_class" in node
            assert "right_class" in node


def test_direct_multiclass_tree_keeps_global_class_semantics_in_descendants() -> None:
    X = np.arange(9, dtype=float).reshape(-1, 1)
    y = np.repeat([0, 1, 2], 3)
    features = np.asarray(["measurement"])

    tree = get_tree(
        {
            "X_train": X,
            "y_train": y,
            "X_test": X,
            "y_test": y,
            "features": features,
            "classes": np.asarray([0, 1, 2]),
            "stat_steps": 1,
            "max_level": 3,
            "split_criterion": "mcc",
        }
    )

    assert all("left_class" in node and "right_class" in node for node in tree)
    assert_array_equal(tree_prediction(tree, X, features), y)


def test_forest_builder_rejects_held_out_data_to_prevent_leakage() -> None:
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.asarray([0, 0, 0, 1, 1, 1])

    with pytest.raises(ValueError, match="Leakage guard"):
        get_forest(
            number_of_trees=1,
            ga_settings={"population_number": 2, "max_epochs": 1, "stop_power": 1.0},
            k=1,
            n=2,
            data={"X_train": X, "y_train": y, "X_test": X, "y_test": y},
            test_size=0.3,
            features=np.asarray(["f0", "f1"]),
            tree_settings={"stat_steps": 1, "max_level": 2, "split_criterion": "mcc"},
            voting="weighted",
            verbose=False,
        )


@pytest.mark.parametrize("y", [np.zeros(4, dtype=int), np.asarray([0, 0, 0, 1])])
def test_internal_ab_split_handles_non_stratifiable_bootstrap_samples(y: np.ndarray) -> None:
    X = np.arange(8, dtype=float).reshape(4, 2)

    X_train, X_test, y_train, y_test = _safe_stratified_split(X, y, test_size=0.5, random_state=3)

    assert X_train.shape == X_test.shape == (2, 2)
    assert y_train.size == y_test.size == 2
