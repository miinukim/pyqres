"""Structural contracts for executable reservoir implementations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .types import BackendLike, FeatureMatrix, InputSequence


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

    def transform(self, inputs: InputSequence) -> FeatureMatrix: ...


@runtime_checkable
class StreamingReservoirProtocol(Protocol):
    """Reservoir exposing the streaming API used by task packages."""

    def run_stream(self, inputs: InputSequence) -> FeatureMatrix: ...


@runtime_checkable
class BatchReservoirProtocol(Protocol):
    """Reservoir exposing a simple batch run method."""

    def run(self, inputs: InputSequence) -> FeatureMatrix: ...


@runtime_checkable
class StepReservoirProtocol(Protocol):
    """Stateful reservoir that can advance one scalar input at a time."""

    def reset(self, *args: Any, **kwargs: Any) -> None: ...

    def step(self, u: float) -> np.ndarray: ...


@runtime_checkable
class QRCReservoirProtocol(StepReservoirProtocol, BatchReservoirProtocol, Protocol):
    """Exact/trajectory QRC frontend with reset, step, and batch execution."""


@runtime_checkable
class StatefulReservoirProtocol(
    StepReservoirProtocol,
    StreamingReservoirProtocol,
    TransformReservoirProtocol,
    Protocol,
):
    """Stateful streaming reservoir used by pyqres experiments."""


@runtime_checkable
class ChannelReservoirProtocol(
    QRCReservoirProtocol, TransformReservoirProtocol, Protocol
):
    """Reservoir exposing its induced memory channel and PTM."""

    def channel(self, u: float, op_memory: np.ndarray) -> np.ndarray: ...

    def ptm(self, u: float) -> np.ndarray: ...


@runtime_checkable
class CircuitReservoirProtocol(StreamingReservoirProtocol, Protocol):
    """Qiskit-style circuit reservoir frontend.

    This matches pyqres.qiskit.QRCReservoir. Circuit reservoirs are not required
    to expose reset/step because their state is represented by the emitted
    circuit rather than a mutable Python density matrix.
    """

    def build_streaming_circuit(
        self, inputs: InputSequence, measure_system: bool = True
    ) -> tuple[Any, list[int], list[int]]: ...

    def build_executable_circuit(
        self,
        inputs: InputSequence,
        backend: BackendLike | None = None,
        measure_system: bool = True,
        optimization_level: int | None = None,
        **transpile_options: Any,
    ) -> Any: ...

    def features_from_counts(
        self,
        counts: Mapping[str, int],
        sys_bits_per_step: list[int],
        anc_bits_per_step: list[int],
    ) -> FeatureMatrix: ...


@runtime_checkable
class QuantumCircuitProtocol(Protocol):
    """Minimal raw-circuit artifact accepted by Qiskit circuit dynamics."""

    num_qubits: int

    def to_instruction(self) -> Any: ...


@runtime_checkable
class SparsePauliOpProtocol(Protocol):
    """Minimal Qiskit-native Hamiltonian artifact for PauliEvolutionGate."""

    num_qubits: int


@runtime_checkable
class QiskitReservoirConfigProtocol(Protocol):
    """Configuration object consumed by the Qiskit reservoir implementation."""

    n_system: int
    n_ancilla: int
    reservoir_type: str
    reservoir_circuit: QuantumCircuitProtocol | None
    reservoir_circuit_targets: tuple[int, ...] | None
    H0_hamiltonian: SparsePauliOpProtocol | None
    H1_hamiltonian: SparsePauliOpProtocol | None
    tau: float
    input_scale: float
    seed: int
    include_bias: bool
    shots: int
    simulator_method: str
    simulator_device: str
    use_noise_model: bool
    aer_options: Mapping[str, Any]

    def total_qubits(self) -> int: ...


@runtime_checkable
class DimensionModelProtocol(Protocol):
    """Dimension-analysis model contract used by memory-observable reservoirs."""

    n_memory: int
    n_readout: int
    dim_memory: int
    dim_readout: int

    def channel(self, u: float, rho: np.ndarray) -> np.ndarray: ...

    def ptm(self, u: float) -> np.ndarray: ...

    def parse_memory_observable(self, spec: str) -> np.ndarray: ...

    def default_memory_observable_specs(
        self,
        preset: str = "z",
        custom_specs: Sequence[str] | None = None,
    ) -> list[str]: ...


@runtime_checkable
class MemoryObservableReservoirProtocol(StatefulReservoirProtocol, Protocol):
    """Reservoir wrapper that emits expectation values of memory observables."""

    model: DimensionModelProtocol
    observables: Sequence[np.ndarray]
    include_bias: bool
    init_state: str
