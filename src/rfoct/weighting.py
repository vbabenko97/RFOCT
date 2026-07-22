"""Analytic-hierarchy-process weights used by RFOCT forest voting."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def rank_quality_levels(powers: Sequence[float]) -> np.ndarray:
    """Return dense quality levels in input order, with larger power ranked higher.

    The historical RFOCT implementation sorts estimator powers in ascending order,
    assigns level 1 to the weakest distinct power, and increments the level for each
    better distinct power. Equal powers share a level.
    """
    values = np.asarray(powers, dtype=float)
    if values.ndim != 1:
        raise ValueError("powers must be one-dimensional")
    if values.size == 0:
        return np.array([], dtype=int)
    if not np.all(np.isfinite(values)):
        raise ValueError("powers must contain only finite values")

    distinct = np.unique(values)
    return np.searchsorted(distinct, values).astype(int) + 1


def ahp_weights(quality_levels: Sequence[float]) -> np.ndarray:
    """Return normalized AHP weights for positive estimator quality levels.

    RFOCT uses the consistent reciprocal comparison matrix ``a_ij = q_i / q_j``.
    Its row geometric means are proportional to ``q_i``; normalization therefore
    yields the same result as dividing every quality level by their sum. Level 1 is
    the weakest estimator and larger levels are better.
    """
    levels = np.asarray(quality_levels, dtype=float)
    if levels.ndim != 1:
        raise ValueError("quality_levels must be one-dimensional")
    if levels.size == 0:
        return np.array([], dtype=float)
    if not np.all(np.isfinite(levels)) or np.any(levels <= 0):
        raise ValueError("quality_levels must be finite and strictly positive")
    return levels / levels.sum()


def weights_from_powers(powers: Sequence[float]) -> np.ndarray:
    """Rank estimator powers and return their AHP voting weights."""
    return ahp_weights(rank_quality_levels(powers))


__all__ = ["ahp_weights", "rank_quality_levels", "weights_from_powers"]
