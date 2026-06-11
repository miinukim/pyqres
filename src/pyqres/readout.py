from __future__ import annotations

"""Readout models used by generic pyqres experiments."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from pyqres.utils.linear import ridge_regression_fit, ridge_regression_predict


class ReadoutModel(Protocol):
    """Minimal readout protocol consumed by Experiment."""

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "ReadoutModel": ...

    def predict(self, features: np.ndarray) -> np.ndarray: ...


@dataclass
class Ridge:
    """Linear ridge readout.

    Reservoirs normally emit a bias column themselves. Set include_bias=True
    only when fitting features that do not already contain one.
    """

    l2: float = 1e-6
    include_bias: bool = False
    weights: np.ndarray | None = None

    def _features(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        if x.ndim != 2:
            raise ValueError(f"features must be a 2D matrix, got shape {x.shape}")
        if self.include_bias:
            x = np.hstack([np.ones((x.shape[0], 1)), x])
        return x

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "Ridge":
        """Fit readout weights."""

        self.weights = ridge_regression_fit(self._features(features), np.asarray(targets, dtype=float), l2=float(self.l2))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict targets from reservoir features."""

        if self.weights is None:
            raise RuntimeError("readout must be fitted before predict")
        return ridge_regression_predict(self._features(features), self.weights)
