"""Genetic feature selection for RFOCT trees."""

# Derived from research code: Copyright © 2019-2023 Vitalii Babenko.
# Historical notice: All rights reserved.

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from copy import deepcopy
from math import ceil

import numpy as np
from sklearn.model_selection import train_test_split

from .classification_evaluation import (
    calculate_mcc_macro,
    calculate_metrics,
    get_confusion_matrix,
)
from .optimal_complexity_tree import get_tree, tree_prediction


def check_equality(population: Sequence[object]) -> bool:
    """Return whether all population members compare equal."""
    return len({tuple(np.asarray(item).tolist()) for item in population}) == 1


def check_duplicates(individual: np.ndarray) -> bool:
    """Return whether a chromosome contains a feature more than once."""
    return individual.size != np.unique(individual).size


def generating_first_population(population_number: int, k: int, n: int) -> np.ndarray:
    """Generate sorted feature-index chromosomes."""
    return np.asarray(
        [sorted(random.sample(range(n), k)) for _ in range(population_number)], dtype=int
    )


def tournament(probabilities: np.ndarray) -> int:
    """Select the fitter of two randomly sampled chromosomes."""
    first, second = random.sample(range(probabilities.size), 2)
    return first if probabilities[first] > probabilities[second] else second


def selection(probabilities: np.ndarray) -> np.ndarray:
    """Select fathers with a degeneracy guard for tiny populations."""
    fathers = np.asarray([tournament(probabilities) for _ in range(probabilities.size)], dtype=int)
    if np.unique(fathers).size == 1 and probabilities.size >= 2:
        return np.arange(probabilities.size)
    return fathers


def get_mothers(fathers: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Select a distinct mate for every father."""
    mothers = []
    for father in fathers:
        mother = tournament(probabilities)
        if probabilities.size >= 2 and mother == father:
            mother = (int(father) + 1) % probabilities.size
        mothers.append(mother)
    return np.asarray(mothers, dtype=int)


def crossover(population: np.ndarray, fathers: np.ndarray, mothers: np.ndarray) -> np.ndarray:
    """Apply the historical uniform crossover in-place."""
    previous = population.copy()
    for index, (father, mother) in enumerate(zip(fathers, mothers, strict=True)):
        for gene in range(population.shape[1]):
            population[index, gene] = (
                previous[mother, gene] if random.random() > 0.5 else previous[father, gene]
            )
    return np.sort(population, axis=1)


def _bootstrap_inbag_oob_indices(
    rng: np.random.Generator, n_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    inbag = rng.integers(0, n_samples, size=n_samples, endpoint=False).astype(int)
    counts = np.bincount(inbag, minlength=n_samples)
    return inbag, np.flatnonzero(counts == 0).astype(int)


def _safe_stratified_split(
    x: np.ndarray, y: np.ndarray, test_size: float, random_state: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    classes, counts = np.unique(y, return_counts=True)
    test_count = ceil(y.size * test_size)
    train_count = y.size - test_count
    can_stratify = (
        classes.size >= 2
        and counts.min() >= 2
        and train_count >= classes.size
        and test_count >= classes.size
    )
    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if can_stratify else None,
    )


def _powers_to_probs(powers: np.ndarray) -> np.ndarray:
    scaled = powers - np.min(powers)
    total = float(np.sum(scaled))
    if not np.isfinite(total) or total <= 0:
        return np.full(powers.size, 1.0 / powers.size)
    return scaled / total


def _score_predictions(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> float:
    if classes.size > 2:
        return calculate_mcc_macro(y_true, y_pred, labels=classes)
    return calculate_metrics(get_confusion_matrix(y_true, y_pred))["mcc"]


def genetic_tree(
    tree_seed: int,
    ga_settings: Mapping[str, object],
    k: int,
    n: int,
    x: np.ndarray,
    y: np.ndarray,
    ab_val_size: float,
    features: np.ndarray,
    tree_settings: Mapping[str, object],
    verbose: bool = True,
) -> dict[str, object]:
    """Select a feature subset with GA fitness measured on OOB samples."""
    random.seed(tree_seed)
    population_number = int(ga_settings["population_number"])
    max_epochs = int(ga_settings["max_epochs"])
    stop_power = float(ga_settings["stop_power"])
    max_level = int(tree_settings["max_level"])
    split_criterion = str(tree_settings.get("split_criterion", "mcc"))
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    classes = np.unique(y)
    population = generating_first_population(population_number, k, n)
    rng = np.random.default_rng(tree_seed)
    best_power = float("-inf")
    best_tree: list[dict[str, object]] = []
    best_features = np.array([], dtype=features.dtype)
    best_oob = np.array([], dtype=int)

    for _ in range(max_epochs):
        powers: list[float] = []
        trees: list[list[dict[str, object]]] = []
        selected_features: list[np.ndarray] = []
        oob_sets: list[np.ndarray] = []
        for individual in population:
            feature_subset = features[individual]
            selected_features.append(feature_subset)
            inbag, oob = _bootstrap_inbag_oob_indices(rng, x.shape[0])
            for _ in range(3):
                if oob.size:
                    break
                inbag, oob = _bootstrap_inbag_oob_indices(rng, x.shape[0])
            x_a, x_b, y_a, y_b = _safe_stratified_split(x[inbag], y[inbag], ab_val_size, tree_seed)
            tree = get_tree(
                {
                    "X_train": x_a[:, individual],
                    "y_train": y_a,
                    "X_test": x_b[:, individual],
                    "y_test": y_b,
                    "features": feature_subset,
                    "max_level": max_level,
                    "split_criterion": split_criterion,
                    "classes": classes,
                }
            )
            trees.append(tree)
            oob_sets.append(oob)
            if oob.size:
                evaluation_x = x[oob][:, individual]
                evaluation_y = y[oob]
            else:
                evaluation_x = x_b[:, individual]
                evaluation_y = y_b
            prediction = tree_prediction(tree, evaluation_x, feature_subset)
            powers.append(_score_predictions(evaluation_y, prediction, classes))

        power_array = np.asarray(powers)
        best_index = int(np.argmax(power_array))
        if power_array[best_index] > best_power:
            best_power = float(power_array[best_index])
            best_tree = deepcopy(trees[best_index])
            best_features = selected_features[best_index].copy()
            best_oob = oob_sets[best_index].copy()
        if best_power >= stop_power:
            break

        order = np.argsort(power_array)[::-1]
        population = population[order]
        probabilities = _powers_to_probs(power_array[order])
        fathers = selection(probabilities)
        population = crossover(population, fathers, get_mothers(fathers, probabilities))
        for index in range(population_number):
            if check_duplicates(population[index]):
                population[index] = sorted(random.sample(range(n), k))
        if check_equality(population):
            for index in range(1, population_number):
                population[index] = sorted(random.sample(range(n), k))

    if verbose:
        print("Best power result: ", best_power)
    return {
        "best_tree": best_tree,
        "best_features": best_features,
        "best_power": best_power,
        "best_oob_idx": best_oob,
    }


__all__ = [
    "check_duplicates",
    "check_equality",
    "crossover",
    "generating_first_population",
    "genetic_tree",
    "get_mothers",
    "selection",
    "tournament",
]
