"""Scikit-learn compatible estimator for RFOCT."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite, sqrt
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from .forest import forest_prediction, get_forest
from .io import load_model_json, save_model_json


class RFOCTClassifier(ClassifierMixin, BaseEstimator):
    """Random Forest of Optimal-Complexity Trees classifier.

    ``predict_proba`` returns normalized weighted vote shares. They are useful as
    ensemble-support scores but are not calibrated posterior probabilities.
    """

    def __init__(
        self,
        n_estimators: int = 51,
        ab_val_size: float = 0.5,
        max_level: int = 5,
        max_features: str | int = "sqrt",
        voting: str = "weighted",
        ga_population: int = 6,
        ga_epochs: int = 10,
        ga_stop_power: float = 1.0,
        random_state: int | None = 0,
        feature_names: list[str] | None = None,
        verbose: int = 0,
        split_criterion: str = "mcc",
    ) -> None:
        self.n_estimators = n_estimators
        self.ab_val_size = ab_val_size
        self.max_level = max_level
        self.max_features = max_features
        self.voting = voting
        self.ga_population = ga_population
        self.ga_epochs = ga_epochs
        self.ga_stop_power = ga_stop_power
        self.random_state = random_state
        self.feature_names = feature_names
        self.verbose = verbose
        self.split_criterion = split_criterion

    def fit(self, X: object, y: object) -> RFOCTClassifier:
        """Fit the forest using only training data."""
        self._validate_hyperparameters()
        x_array, y_array = validate_data(self, X, y, dtype="numeric", ensure_min_samples=4)
        check_classification_targets(y_array)
        feature_names = self._infer_feature_names(X)
        if len(feature_names) != x_array.shape[1]:
            raise ValueError("feature_names length must match the number of X features")

        classes, encoded_y = np.unique(y_array, return_inverse=True)
        if classes.size < 2:
            raise ValueError("RFOCTClassifier requires at least two target classes")

        n_features = x_array.shape[1]
        k = self._resolve_max_features(n_features)
        seed_offset = (
            int(np.random.default_rng().integers(0, 1_000_000_000))
            if self.random_state is None
            else self.random_state * 100_000
        )
        self.forest_ = get_forest(
            number_of_trees=self.n_estimators,
            ga_settings={
                "population_number": self.ga_population,
                "max_epochs": self.ga_epochs,
                "stop_power": self.ga_stop_power,
            },
            k=k,
            n=n_features,
            data={"X_train": x_array, "y_train": encoded_y},
            test_size=self.ab_val_size,
            features=np.asarray(feature_names),
            tree_settings={
                "max_level": self.max_level,
                "split_criterion": self.split_criterion,
            },
            voting=self.voting,
            seed_offset=seed_offset,
            verbose=bool(self.verbose),
        )
        self.classes_ = classes
        self.n_features_in_ = n_features
        self.rfoct_feature_names_ = np.asarray(feature_names, dtype=object)
        self._index_to_class_ = {index: label for index, label in enumerate(classes)}
        return self

    def predict(self, X: object) -> np.ndarray:
        """Predict class labels."""
        output = self._predict_encoded(X)
        return np.asarray([self._index_to_class_[int(label)] for label in output["y_pred"]])

    def predict_proba(self, X: object) -> np.ndarray:
        """Return normalized class vote shares in ``classes_`` order."""
        output = self._predict_encoded(X)
        if len(self.classes_) > 2:
            scores = np.asarray(output["y_proba"], dtype=float)
            totals = scores.sum(axis=1, keepdims=True)
            return np.divide(scores, totals, out=np.zeros_like(scores), where=totals != 0)
        positive = np.asarray(output["y_proba"], dtype=float).reshape(-1, 1)
        return np.concatenate([1.0 - positive, positive], axis=1)

    def decision_function(self, X: object) -> np.ndarray:
        """Return signed binary support or multiclass vote shares."""
        vote_shares = self.predict_proba(X)
        if vote_shares.shape[1] != 2:
            return vote_shares
        margin = vote_shares[:, 1] - 0.5
        margin[margin == 0] = np.finfo(float).eps
        return margin

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible fitted-model payload."""
        check_is_fitted(self, attributes=["forest_", "classes_", "rfoct_feature_names_"])
        return {
            "type": "RFOCTClassifier",
            "version": 1,
            "params": self.get_params(deep=False),
            "feature_names": self.rfoct_feature_names_.tolist(),
            "classes_original": self.classes_.tolist(),
            "label_encoding": "ordinal_0_to_k_minus_1",
            "forest": self.forest_,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RFOCTClassifier:
        """Restore a fitted estimator from ``to_dict`` output."""
        if payload.get("type") != "RFOCTClassifier":
            raise ValueError("Unsupported model payload")
        params_value = payload.get("params", {})
        if not isinstance(params_value, Mapping):
            raise ValueError("Model params must be a mapping")
        params = dict(params_value)
        params.pop("stat_steps", None)
        model = cls(**params)

        forest = payload.get("forest")
        feature_names = payload.get("feature_names")
        classes = payload.get("classes_original")
        if (
            not isinstance(forest, list)
            or not isinstance(feature_names, list)
            or not isinstance(classes, list)
        ):
            raise ValueError("Model payload is missing fitted forest metadata")
        model.forest_ = forest
        model.rfoct_feature_names_ = np.asarray(feature_names, dtype=object)
        model.classes_ = np.asarray(classes)
        model.n_features_in_ = len(feature_names)
        model._index_to_class_ = {index: label for index, label in enumerate(model.classes_)}
        return model

    def save(self, path: str | Path) -> None:
        """Serialize a fitted estimator to JSON."""
        save_model_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> RFOCTClassifier:
        """Load an estimator saved by ``save``."""
        return cls.from_dict(load_model_json(path))

    def _predict_encoded(self, X: object) -> dict[str, object]:
        check_is_fitted(self, attributes=["forest_", "classes_", "rfoct_feature_names_"])
        x_array = self._prepare_prediction_data(X)
        return forest_prediction(self.forest_, x_array, self.rfoct_feature_names_)

    def _prepare_prediction_data(self, X: object) -> np.ndarray:
        if hasattr(X, "loc") and hasattr(X, "columns"):
            try:
                X = X.loc[:, self.rfoct_feature_names_.tolist()]
            except KeyError as error:
                raise ValueError("X is missing one or more fitted features") from error
        return validate_data(self, X, dtype="numeric", reset=False)

    def _infer_feature_names(self, X: object) -> list[str]:
        if self.feature_names is not None:
            names = list(self.feature_names)
        elif hasattr(X, "columns"):
            names = [str(name) for name in X.columns]
        else:
            shape = getattr(X, "shape", None)
            if not isinstance(shape, Sequence) or len(shape) != 2:
                array = np.asarray(X)
                if array.ndim != 2:
                    raise ValueError("X must be a two-dimensional feature matrix")
                n_features = array.shape[1]
            else:
                n_features = int(shape[1])
            names = [f"f{index}" for index in range(n_features)]
        if not names or len(names) != len(set(names)):
            raise ValueError("feature_names must be non-empty and unique")
        return names

    def _resolve_max_features(self, n_features: int) -> int:
        if self.max_features == "sqrt":
            return max(1, int(round(sqrt(n_features))))
        if isinstance(self.max_features, bool) or not isinstance(self.max_features, int):
            raise ValueError("max_features must be 'sqrt' or an integer")
        if not 1 <= self.max_features <= n_features:
            raise ValueError("max_features must be between 1 and the number of X features")
        return self.max_features

    def _validate_hyperparameters(self) -> None:
        integer_parameters = {
            "n_estimators": (self.n_estimators, 1),
            "max_level": (self.max_level, 1),
            "ga_population": (self.ga_population, 2),
            "ga_epochs": (self.ga_epochs, 1),
        }
        for name, (value, minimum) in integer_parameters.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
        if not 0.0 < self.ab_val_size < 1.0:
            raise ValueError("ab_val_size must be strictly between 0 and 1")
        if self.voting not in {"weighted", "uniform"}:
            raise ValueError("voting must be 'weighted' or 'uniform'")
        if self.split_criterion not in {"mcc", "gini"}:
            raise ValueError("split_criterion must be 'mcc' or 'gini'")
        if not isfinite(self.ga_stop_power):
            raise ValueError("ga_stop_power must be finite")
        if self.random_state is not None and (
            isinstance(self.random_state, bool)
            or not isinstance(self.random_state, int)
            or self.random_state < 0
        ):
            raise ValueError("random_state must be a non-negative integer or None")
        if isinstance(self.verbose, bool) or not isinstance(self.verbose, int) or self.verbose < 0:
            raise ValueError("verbose must be a non-negative integer")


__all__ = ["RFOCTClassifier"]
