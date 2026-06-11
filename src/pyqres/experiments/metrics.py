from __future__ import annotations

"""Generic metrics for reservoir readout experiments."""

from typing import Callable, Mapping

import numpy as np


Metric = Callable[[np.ndarray, np.ndarray], float]


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""

    y = np.asarray(y_true)
    yhat = np.asarray(y_pred)
    return float(np.mean((y - yhat) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""

    return float(np.sqrt(mse(y_true, y_pred)))


def negative_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Negative RMSE for score-style optimization."""

    return -rmse(y_true, y_pred)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination."""

    y = np.asarray(y_true).reshape(-1)
    yhat = np.asarray(y_pred).reshape(-1)
    denom = np.var(y)
    if denom <= 1e-12:
        return 0.0
    return float(1.0 - np.mean((y - yhat) ** 2) / denom)


def resolve_metrics(metrics: Mapping[str, Metric] | list[str] | tuple[str, ...] | None) -> dict[str, Metric]:
    """Normalize metric names/callables into a dictionary."""

    registry: dict[str, Metric] = {
        "mse": mse,
        "rmse": rmse,
        "negative_rmse": negative_rmse,
        "r2": r2,
    }
    if metrics is None:
        return {"r2": r2, "mse": mse}
    if isinstance(metrics, Mapping):
        return dict(metrics)
    out: dict[str, Metric] = {}
    for name in metrics:
        key = str(name)
        if key not in registry:
            raise ValueError(f"Unknown metric '{key}'. Available metrics: {sorted(registry)}")
        out[key] = registry[key]
    return out
