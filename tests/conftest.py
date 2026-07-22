from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from rfoct import RFOCTClassifier


@pytest.fixture(scope="session")
def binary_data() -> tuple[np.ndarray, np.ndarray]:
    X, y_encoded = make_classification(
        n_samples=48,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=1.8,
        random_state=11,
    )
    y = np.where(y_encoded == 0, "healthy", "pathology")
    return X, y


@pytest.fixture(scope="session")
def multiclass_data() -> tuple[np.ndarray, np.ndarray]:
    X, y_encoded = make_classification(
        n_samples=72,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=23,
    )
    y = np.asarray([f"class-{value}" for value in y_encoded])
    return X, y


@pytest.fixture(scope="session")
def estimator_params() -> dict[str, object]:
    return {
        "n_estimators": 2,
        "ab_val_size": 0.3,
        "max_level": 3,
        "max_features": 2,
        "ga_population": 2,
        "ga_epochs": 1,
        "random_state": 37,
        "verbose": 0,
    }


@pytest.fixture(scope="session")
def binary_model(
    binary_data: tuple[np.ndarray, np.ndarray],
    estimator_params: dict[str, object],
) -> RFOCTClassifier:
    X, y = binary_data
    return RFOCTClassifier(**estimator_params).fit(X, y)


@pytest.fixture(scope="session")
def multiclass_model(
    multiclass_data: tuple[np.ndarray, np.ndarray],
    estimator_params: dict[str, object],
) -> RFOCTClassifier:
    X, y = multiclass_data
    return RFOCTClassifier(**estimator_params).fit(X, y)
