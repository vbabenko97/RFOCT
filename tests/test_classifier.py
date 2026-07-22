from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone, is_classifier
from sklearn.datasets import make_regression
from sklearn.exceptions import NotFittedError
from sklearn.utils import get_tags

from rfoct import RFOCTClassifier


def test_binary_fit_predict_and_probability_contract(
    binary_model: RFOCTClassifier,
    binary_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = binary_data

    prediction = binary_model.predict(X)
    probability = binary_model.predict_proba(X)

    assert_array_equal(binary_model.classes_, np.unique(y))
    assert prediction.shape == y.shape
    assert set(prediction) <= set(binary_model.classes_)
    assert probability.shape == (X.shape[0], 2)
    assert_allclose(probability.sum(axis=1), 1.0)
    assert np.all((probability >= 0.0) & (probability <= 1.0))
    assert_array_equal(
        (binary_model.decision_function(X) > 0).astype(int),
        (prediction == binary_model.classes_[1]).astype(int),
    )


def test_multiclass_fit_predict_and_probability_contract(
    multiclass_model: RFOCTClassifier,
    multiclass_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = multiclass_data

    prediction = multiclass_model.predict(X)
    probability = multiclass_model.predict_proba(X)

    assert_array_equal(multiclass_model.classes_, np.unique(y))
    assert prediction.shape == y.shape
    assert set(prediction) <= set(multiclass_model.classes_)
    assert probability.shape == (X.shape[0], 3)
    assert_allclose(probability.sum(axis=1), 1.0)
    assert np.all((probability >= 0.0) & (probability <= 1.0))


def test_same_random_state_reproduces_forest_predictions_and_probabilities(
    tmp_path: Path,
    binary_data: tuple[np.ndarray, np.ndarray],
    estimator_params: dict[str, object],
) -> None:
    X, y = binary_data
    first = RFOCTClassifier(**estimator_params).fit(X, y)
    second = RFOCTClassifier(**estimator_params).fit(X, y)
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first.save(first_path)
    second.save(second_path)

    assert json.loads(first_path.read_text(encoding="utf-8")) == json.loads(
        second_path.read_text(encoding="utf-8")
    )
    assert_array_equal(first.predict(X), second.predict(X))
    assert_allclose(first.predict_proba(X), second.predict_proba(X))


def test_estimator_can_be_cloned_and_parameters_can_be_updated(
    estimator_params: dict[str, object],
) -> None:
    estimator = RFOCTClassifier(**estimator_params)

    cloned = clone(estimator)

    assert cloned.get_params(deep=False) == estimator.get_params(deep=False)
    assert estimator.set_params(n_estimators=3) is estimator
    assert estimator.n_estimators == 3


def test_estimator_exposes_classifier_tags() -> None:
    estimator = RFOCTClassifier()

    assert is_classifier(estimator)
    assert get_tags(estimator).estimator_type == "classifier"


@pytest.mark.parametrize("method_name", ["predict", "predict_proba"])
def test_prediction_before_fit_raises_not_fitted_error(method_name: str) -> None:
    estimator = RFOCTClassifier()

    with pytest.raises(NotFittedError):
        getattr(estimator, method_name)(np.zeros((2, 2)))


def test_predict_rejects_wrong_number_of_features(
    binary_model: RFOCTClassifier,
    binary_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, _ = binary_data

    with pytest.raises(ValueError, match="features"):
        binary_model.predict(X[:, :-1])


def test_fit_rejects_continuous_regression_targets(
    estimator_params: dict[str, object],
) -> None:
    X, y = make_regression(n_samples=12, n_features=3, random_state=5)

    with pytest.raises(ValueError, match="continuous"):
        RFOCTClassifier(**estimator_params).fit(X, y)


def test_json_serialization_round_trip_preserves_public_behavior(
    tmp_path: Path,
    binary_model: RFOCTClassifier,
    binary_data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, _ = binary_data
    path = tmp_path / "rfoct.json"

    binary_model.save(path)
    restored = RFOCTClassifier.load(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["type"] == "RFOCTClassifier"
    assert restored.get_params(deep=False) == binary_model.get_params(deep=False)
    assert_array_equal(restored.classes_, binary_model.classes_)
    assert_array_equal(restored.predict(X), binary_model.predict(X))
    assert_allclose(restored.predict_proba(X), binary_model.predict_proba(X))
