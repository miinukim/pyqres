from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, TypeAlias, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class ReservoirStepResult:
    """One reservoir step result in the unified pyqres interface."""

    features: np.ndarray
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ReservoirRunResult:
    """Batch reservoir run result in the unified pyqres interface."""

    features: np.ndarray
    metadata: dict[str, Any] | None = None


ConfigMapping: TypeAlias = Mapping[str, Any]
MetricCallable: TypeAlias = Callable[[np.ndarray, np.ndarray], float]


@runtime_checkable
class QRCReservoirProtocol(Protocol):
    """Minimal protocol shared by exact, Qiskit, and trajectory reservoirs."""

    def reset(self) -> None:
        ...

    def step(self, u: float) -> np.ndarray:
        ...

    def run(self, inputs: Sequence[float] | np.ndarray) -> np.ndarray:
        ...


@runtime_checkable
class TransformReservoirProtocol(Protocol):
    """Scikit-style reservoir transformer used by generic experiments."""

    def transform(self, inputs: Sequence[float] | np.ndarray) -> np.ndarray:
        ...


@runtime_checkable
class StatefulReservoirProtocol(TransformReservoirProtocol, Protocol):
    """Reservoir that exposes explicit recurrent state control."""

    def reset(self, *args: Any, **kwargs: Any) -> None:
        ...

    def step(self, u: float) -> np.ndarray:
        ...


@runtime_checkable
class ChannelReservoirProtocol(QRCReservoirProtocol, Protocol):
    """Reservoir protocol for models that expose their induced memory channel."""

    def channel(self, u: float, op_memory: np.ndarray) -> np.ndarray:
        ...

    def ptm(self, u: float) -> np.ndarray:
        ...


@runtime_checkable
class CircuitReservoirProtocol(QRCReservoirProtocol, Protocol):
    """Reservoir protocol for implementations that can emit Qiskit circuits."""

    def circuit(self, inputs: Sequence[float] | np.ndarray, **kwargs: Any) -> Any:
        ...


@runtime_checkable
class DatasetProtocol(Protocol):
    """Dataset contract consumed by task-agnostic experiments."""

    inputs: np.ndarray
    targets: np.ndarray
    metadata: Mapping[str, Any] | None

    def validate_features(self, features: np.ndarray) -> None:
        ...

    def save_npz(self, path: str | Path) -> Path:
        ...


@runtime_checkable
class ReadoutProtocol(Protocol):
    """Supervised readout contract used by Experiment."""

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "ReadoutProtocol":
        ...

    def predict(self, features: np.ndarray) -> np.ndarray:
        ...


@runtime_checkable
class SerializableSpecProtocol(Protocol):
    """Protocol for public specs that round-trip through dictionaries."""

    def to_dict(self) -> dict[str, Any]:
        ...


@runtime_checkable
class ExperimentResultProtocol(Protocol):
    """Persistable result produced by experiment runners."""

    metrics: Mapping[str, Mapping[str, float]]
    features: np.ndarray
    predictions: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]

    def save(self, outdir: str | Path) -> Path:
        ...
