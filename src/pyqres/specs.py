from __future__ import annotations

"""Serializable public specifications for constructing reservoirs."""

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ReadoutSpec:
    """Observable/readout configuration for reservoir feature extraction."""

    mode: str = "ancilla_probs"
    observables: str | Sequence[str] = "z"
    count: int | None = None
    custom: tuple[str, ...] = ()
    include_bias: bool = True
    init_state: str = "maximally_mixed"
    use_shot_noise: bool = False
    shots: int = 4096


@dataclass(frozen=True)
class ReservoirSpec:
    """Task-agnostic reservoir construction spec.

    Parameters intentionally stay close to the existing implementation while
    keeping room for arbitrary Hamiltonian objects through hamiltonian_kwargs.
    """

    family: str = "ising"
    n_system: int | None = None
    n_ancilla: int | None = None
    n_memory: int | None = None
    n_readout: int | None = None
    tau: float = 1.0
    input_scale: float = 1.0
    seed: int = 17462
    readout: ReadoutSpec = field(default_factory=ReadoutSpec)
    model_kwargs: Mapping[str, Any] = field(default_factory=dict)
    hamiltonian_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def with_updates(self, **updates: Any) -> "ReservoirSpec":
        """Return a copy with selected fields replaced."""

        return replace(self, **updates)

    @property
    def system_qubits(self) -> int:
        """Number of recurrent memory/system qubits."""

        value = self.n_system if self.n_system is not None else self.n_memory
        if value is None:
            raise ValueError("ReservoirSpec requires n_system or n_memory")
        return int(value)

    @property
    def ancilla_qubits(self) -> int:
        """Number of readout/ancilla qubits."""

        value = self.n_ancilla if self.n_ancilla is not None else self.n_readout
        if value is None:
            raise ValueError("ReservoirSpec requires n_ancilla or n_readout")
        return int(value)
