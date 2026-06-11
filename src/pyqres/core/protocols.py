from __future__ import annotations

"""Public structural contracts for pyqres.

The concrete package is intentionally lightweight and duck-typed: reservoirs can
come from dense simulation, dimension-analysis models, Qiskit circuits, or user
objects. These protocols document the stable shapes that pyqres builders,
experiments, and task packages consume without forcing a common base class.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence, TypeAlias, runtime_checkable

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


@dataclass(frozen=True)
class ReservoirStepResult:
    """One optional rich reservoir step result."""

    features: np.ndarray
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ReservoirRunResult:
    """One optional rich reservoir run result."""

    features: np.ndarray
    metadata: dict[str, Any] | None = None


@runtime_checkable
class TransformReservoirProtocol(Protocol):
    """Reservoir exposing the scikit-style transformer API used by Experiment."""

    def transform(self, inputs: InputSequence) -> FeatureMatrix:
        ...


@runtime_checkable
class StreamingReservoirProtocol(Protocol):
    """Reservoir exposing the streaming API used by task packages."""

    def run_stream(self, inputs: InputSequence) -> FeatureMatrix:
        ...


@runtime_checkable
class BatchReservoirProtocol(Protocol):
    """Reservoir exposing a simple batch run method."""

    def run(self, inputs: InputSequence) -> FeatureMatrix:
        ...


@runtime_checkable
class StepReservoirProtocol(Protocol):
    """Stateful reservoir that can advance one scalar input at a time."""

    def reset(self, *args: Any, **kwargs: Any) -> None:
        ...

    def step(self, u: float) -> np.ndarray:
        ...


@runtime_checkable
class QRCReservoirProtocol(StepReservoirProtocol, BatchReservoirProtocol, Protocol):
    """Exact/trajectory QRC frontend with reset, step, and batch execution."""


@runtime_checkable
class StatefulReservoirProtocol(StepReservoirProtocol, StreamingReservoirProtocol, TransformReservoirProtocol, Protocol):
    """Stateful streaming reservoir used by pyqres experiments."""


@runtime_checkable
class ChannelReservoirProtocol(QRCReservoirProtocol, TransformReservoirProtocol, Protocol):
    """Reservoir exposing its induced memory channel and PTM."""

    def channel(self, u: float, op_memory: np.ndarray) -> np.ndarray:
        ...

    def ptm(self, u: float) -> np.ndarray:
        ...


@runtime_checkable
class CircuitReservoirProtocol(StreamingReservoirProtocol, Protocol):
    """Qiskit-style circuit reservoir frontend.

    This matches pyqres.qiskit.QRCReservoir. Circuit reservoirs are not required
    to expose reset/step because their state is represented by the emitted
    circuit rather than a mutable Python density matrix.
    """

    def build_streaming_circuit(self, inputs: InputSequence, measure_system: bool = True) -> tuple[Any, list[int], list[int]]:
        ...

    def build_executable_circuit(
        self,
        inputs: InputSequence,
        backend: BackendLike | None = None,
        measure_system: bool = True,
        optimization_level: int | None = None,
        **transpile_options: Any,
    ) -> Any:
        ...

    def features_from_counts(self, counts: Mapping[str, int], sys_bits_per_step: list[int], anc_bits_per_step: list[int]) -> FeatureMatrix:
        ...


@runtime_checkable
class DimensionModelProtocol(Protocol):
    """Dimension-analysis model contract used by memory-observable reservoirs."""

    n_memory: int
    n_readout: int
    dim_memory: int
    dim_readout: int

    def channel(self, u: float, rho: np.ndarray) -> np.ndarray:
        ...

    def ptm(self, u: float) -> np.ndarray:
        ...

    def parse_memory_observable(self, spec: str) -> np.ndarray:
        ...

    def default_memory_observable_specs(
        self,
        preset: str = "z",
        custom_specs: Sequence[str] | None = None,
    ) -> list[str]:
        ...


@runtime_checkable
class MemoryObservableReservoirProtocol(StatefulReservoirProtocol, Protocol):
    """Reservoir wrapper that emits expectation values of memory observables."""

    model: DimensionModelProtocol
    observables: Sequence[np.ndarray]
    include_bias: bool
    init_state: str


@runtime_checkable
class HamiltonianSpecProtocol(Protocol):
    """Backend-neutral Hamiltonian component."""

    kind: str
    n_qubits: int
    data: Any
    terms: Sequence[Any]

    def to_dense(self) -> np.ndarray:
        ...

    def to_sparse_pauli_op(self) -> Any:
        ...


@runtime_checkable
class ReadoutSpecProtocol(Protocol):
    """Serializable reservoir feature-readout configuration."""

    mode: str
    observables: ObservableSpec
    count: int | None
    custom: Sequence[str]
    include_bias: bool
    init_state: str
    use_shot_noise: bool
    shots: int

    def to_dict(self) -> dict[str, Any]:
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
    preset: str | None
    source_kind: str
    n_system: int | None
    n_ancilla: int | None
    n_memory: int | None
    n_readout: int | None
    tau: float
    input_scale: float
    seed: int
    readout: ReadoutSpecProtocol
    model_kwargs: Mapping[str, Any]
    hamiltonian_kwargs: Mapping[str, Any]
    circuit_kwargs: Mapping[str, Any]
    runtime: Mapping[str, Any]

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
    """Chainable builder used by qres.reservoir(...)."""

    @property
    def spec(self) -> ReservoirSpecProtocol:
        ...

    def memory_qubits(self, n_qubits: int) -> "ReservoirBuilderProtocol":
        ...

    def readout_qubits(self, n_qubits: int) -> "ReservoirBuilderProtocol":
        ...

    def seed(self, value: int) -> "ReservoirBuilderProtocol":
        ...

    def preset(self, name: str, **kwargs: Any) -> "ReservoirBuilderProtocol":
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
        preset: ObservableSpec = "z",
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

    def hamiltonian(self, *args: Any, **kwargs: Any) -> "ReservoirBuilderProtocol":
        ...

    def circuit(self, circuit: CircuitLike, **kwargs: Any) -> "ReservoirBuilderProtocol":
        ...

    def use(self, reservoir: Any) -> "ReservoirBuilderProtocol":
        ...

    def backend(self, name: str = "exact") -> Any:
        ...

    def build(self, backend: str | None = None) -> Any:
        ...


@runtime_checkable
class ReservoirCompilerProtocol(Protocol):
    """Callable object/function that compiles a spec into an executable reservoir."""

    def __call__(self, spec: ReservoirSpecProtocol, backend: str = "exact") -> Any:
        ...


@runtime_checkable
class ReservoirFactoryProtocol(Protocol):
    """Dictionary-first reservoir factory contract."""

    @classmethod
    def builder_from_dict(cls, config: Mapping[str, Any]) -> ReservoirBuilderProtocol:
        ...

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> Any:
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
        indices: DatasetSplitProtocol | Mapping[str, IndexSequence] | None = None,
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


@runtime_checkable
class ReadoutProtocol(Protocol):
    """Supervised readout contract used by Experiment."""

    def fit(self, features: FeatureMatrix, targets: TargetArray) -> "ReadoutProtocol":
        ...

    def predict(self, features: FeatureMatrix) -> np.ndarray:
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
class TaskDatasetFactoryProtocol(Protocol):
    """Factory contract used by pyqres-tasks adapters."""

    def __call__(self, cfg: Any) -> DatasetProtocol:
        ...


@runtime_checkable
class TaskRunnerProtocol(Protocol):
    """Legacy task-runner contract kept for task packages that stream directly."""

    res: StreamingReservoirProtocol
    cfg: Any

    def run(self) -> Mapping[str, float] | Mapping[int, Mapping[str, float]]:
        ...
