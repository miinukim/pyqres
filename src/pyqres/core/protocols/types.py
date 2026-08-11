"""Shared type aliases used by pyqres structural contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np

ConfigMapping: TypeAlias = Mapping[str, Any]
MetricCallable: TypeAlias = Callable[[np.ndarray, np.ndarray], float]
FeatureMatrix: TypeAlias = np.ndarray
TargetArray: TypeAlias = np.ndarray
InputSequence: TypeAlias = Sequence[float] | np.ndarray
ObservableSpec: TypeAlias = str | Sequence[str]
IndexSequence: TypeAlias = Sequence[int] | np.ndarray
PauliTermLike: TypeAlias = Any
HamiltonianLike: TypeAlias = Any
CircuitLike: TypeAlias = Any
BackendLike: TypeAlias = Any
DynamicsLike: TypeAlias = Mapping[str, Any] | tuple[Any, Any] | CircuitLike | Any
QiskitArtifactMap: TypeAlias = Mapping[str, Any]
