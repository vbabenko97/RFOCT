"""JSON serialization helpers for fitted RFOCT estimators."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _to_builtin(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_to_builtin(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_builtin(item) for item in value]
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def save_model_json(path: str | Path, payload: dict[str, object]) -> None:
    """Save a model payload as human-readable JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(_to_builtin(payload), indent=2), encoding="utf-8")


def load_model_json(path: str | Path) -> dict[str, object]:
    """Load a model payload from JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Model payload must be a JSON object")
    return payload


__all__ = ["load_model_json", "save_model_json"]
