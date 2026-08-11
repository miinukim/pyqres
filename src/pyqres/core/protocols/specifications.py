"""Structural contracts for serializable reservoir specifications."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .types import ObservableSpec


@runtime_checkable
class HamiltonianSpecProtocol(Protocol):
    """Backend-neutral Hamiltonian component."""

    kind: str
    n_qubits: int
    data: Any
    terms: Sequence[Any]

    def to_dense(self) -> np.ndarray: ...

    def to_sparse_pauli_op(self) -> Any: ...


@runtime_checkable
class InputEncodingSpecProtocol(Protocol):
    """Serializable description of how inputs are encoded into reservoir steps."""

    mode: str
    operator: str | None
    targets: Sequence[int]
    scale: float
    bias: float
    parameters: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]: ...


@runtime_checkable
class DynamicsSpecProtocol(Protocol):
    """Serializable description of reservoir dynamics independent of presets."""

    kind: str
    name: str | None
    parameters: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]: ...


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

    def to_dict(self) -> dict[str, Any]: ...


@runtime_checkable
class SerializableSpecProtocol(Protocol):
    """Protocol for public specs that round-trip through dictionaries."""

    def to_dict(self) -> dict[str, Any]: ...


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
    encoding: InputEncodingSpecProtocol
    dynamics: DynamicsSpecProtocol
    readout: ReadoutSpecProtocol
    model_kwargs: Mapping[str, Any]
    hamiltonian_kwargs: Mapping[str, Any]
    circuit_kwargs: Mapping[str, Any]
    qiskit_kwargs: Mapping[str, Any]
    runtime: Mapping[str, Any]

    @property
    def system_qubits(self) -> int: ...

    @property
    def ancilla_qubits(self) -> int: ...

    def with_updates(self, **updates: Any) -> ReservoirSpecProtocol: ...
