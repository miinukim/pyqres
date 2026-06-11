from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence, TypeAlias, runtime_checkable

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
FeatureMatrix: TypeAlias = np.ndarray
TargetArray: TypeAlias = np.ndarray
InputSequence: TypeAlias = Sequence[float] | np.ndarray


@runtime_checkable
class QRCReservoirProtocol(Protocol):
    """Minimal protocol shared by exact, Qiskit, and trajectory reservoirs."""

    def reset(self) -> None:
        ...

    def step(self, u: float) -> np.ndarray:
        ...

    def run(self, inputs: InputSequence) -> FeatureMatrix:
        ...


@runtime_checkable
class TransformReservoirProtocol(Protocol):
    """Scikit-style reservoir transformer used by generic experiments."""

    def transform(self, inputs: InputSequence) -> FeatureMatrix:
        ...


@runtime_checkable
class StatefulReservoirProtocol(TransformReservoirProtocol, Protocol):
    """Reservoir that exposes explicit recurrent state control."""

    def reset(self, *args: Any, **kwargs: Any) -> None:
        ...

    def step(self, u: float) -> np.ndarray:
        ...

    def run_stream(self, inputs: InputSequence) -> FeatureMatrix:
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

    def circuit(self, inputs: InputSequence, **kwargs: Any) -> Any:
        ...


@runtime_checkable
class DatasetSplitProtocol(Protocol):
    """Index split contract used by supervised experiments."""

    washout: np.ndarray
    train: np.ndarray
    test: np.ndarray

    def validate(self, n_samples: int) -> None:
        ...

    def to_dict(self) -> dict[str, list[int]]:
        ...


@runtime_checkable
class DatasetProtocol(Protocol):
    """Dataset contract consumed by task-agnostic experiments."""

    inputs: np.ndarray
    targets: np.ndarray
    split: DatasetSplitProtocol
    metadata: Mapping[str, Any] | None

    def validate_features(self, features: np.ndarray) -> None:
        ...

    def save_npz(self, path: str | Path) -> Path:
        ...


@runtime_checkable
class ReadoutProtocol(Protocol):
    """Supervised readout contract used by Experiment."""

    def fit(self, features: FeatureMatrix, targets: TargetArray) -> "ReadoutProtocol":
        ...

    def predict(self, features: FeatureMatrix) -> np.ndarray:
        ...


@runtime_checkable
class SerializableSpecProtocol(Protocol):
    """Protocol for public specs that round-trip through dictionaries."""

    def to_dict(self) -> dict[str, Any]:
        ...


@runtime_checkable
class ReservoirSpecProtocol(SerializableSpecProtocol, Protocol):
    """Reservoir construction spec consumed by compile/build helpers."""

    family: str
    n_system: int | None
    n_ancilla: int | None
    n_memory: int | None
    n_readout: int | None
    tau: float
    input_scale: float
    seed: int
    model_kwargs: Mapping[str, Any]
    hamiltonian_kwargs: Mapping[str, Any]

    @property
    def system_qubits(self) -> int:
        ...

    @property
    def ancilla_qubits(self) -> int:
        ...

    def with_updates(self, **updates: Any) -> "ReservoirSpecProtocol":
        ...


@runtime_checkable
class ReservoirBuilderProtocol(Protocol):
    """Chainable reservoir builder used by the fluent API."""

    @property
    def spec(self) -> ReservoirSpecProtocol:
        ...

    def memory_qubits(self, n_qubits: int) -> "ReservoirBuilderProtocol":
        ...

    def readout_qubits(self, n_qubits: int) -> "ReservoirBuilderProtocol":
        ...

    def seed(self, value: int) -> "ReservoirBuilderProtocol":
        ...

    def input(
        self,
        axis: str = "Z",
        *,
        site: int = 0,
        sites: Sequence[int] | None = None,
        strength: float = 1.0,
        on_memory: bool = True,
        scale: float | None = None,
        normalization: str = "none",
    ) -> "ReservoirBuilderProtocol":
        ...

    def evolution(self, *, tau: float | None = None, **kwargs: Any) -> "ReservoirBuilderProtocol":
        ...

    def observables(
        self,
        preset: str | Sequence[str] = "z",
        *,
        count: int | None = None,
        custom: Sequence[str] | None = None,
        include_bias: bool = True,
        init_state: str = "zero",
    ) -> "ReservoirBuilderProtocol":
        ...

    def ancilla_probabilities(
        self,
        *,
        include_bias: bool = True,
        init_state: str = "maximally_mixed",
        shot_noise: bool = False,
        shots: int = 4096,
    ) -> "ReservoirBuilderProtocol":
        ...

    def model(self, **kwargs: Any) -> "ReservoirBuilderProtocol":
        ...

    def hamiltonian(self, **kwargs: Any) -> "ReservoirBuilderProtocol":
        ...

    def backend(self, name: str = "exact") -> Any:
        ...

    def build(self, backend: str | None = None) -> Any:
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


@runtime_checkable
class ExperimentProtocol(Protocol):
    """Runnable supervised reservoir experiment."""

    reservoir: Any
    dataset: DatasetProtocol
    readout: ReadoutProtocol | None
    metrics: Mapping[str, MetricCallable] | list[str] | tuple[str, ...] | None
    metadata: Mapping[str, Any] | None

    def run(self) -> ExperimentResultProtocol:
        ...


@runtime_checkable
class SweepResultProtocol(Protocol):
    """Persistable collection of sweep experiment results."""

    parameter: str

    def rows(self) -> list[dict[str, Any]]:
        ...

    def save(self, outdir: str | Path) -> Path:
        ...


@runtime_checkable
class SweepProtocol(Protocol):
    """One-parameter reservoir sweep contract."""

    base: ReservoirSpecProtocol
    parameter: str
    values: Iterable[float]

    def specs(self) -> list[ReservoirSpecProtocol]:
        ...

    def run(
        self,
        dataset: DatasetProtocol,
        *,
        backend: str = "exact",
        readout_factory: Callable[[], ReadoutProtocol] | None = None,
        metrics: Mapping[str, MetricCallable] | list[str] | tuple[str, ...] | None = None,
    ) -> SweepResultProtocol:
        ...


@runtime_checkable
class SupervisedDataBuilderProtocol(Protocol):
    """Deferred dataset builder for supervised arrays."""

    inputs: np.ndarray
    targets: np.ndarray
    metadata: Mapping[str, Any]

    def split(
        self,
        *,
        washout: int = 0,
        train: int,
        test: int,
        indices: DatasetSplitProtocol | Mapping[str, Sequence[int]] | None = None,
    ) -> DatasetProtocol:
        ...


@runtime_checkable
class TimeSeriesDataBuilderProtocol(Protocol):
    """Deferred dataset builder for scalar forecasting series."""

    series: np.ndarray
    target_horizon: int
    metadata: Mapping[str, Any]

    def split(self, *, washout: int = 0, train: int, test: int) -> DatasetProtocol:
        ...
