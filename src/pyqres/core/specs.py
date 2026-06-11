from __future__ import annotations

"""Serializable public specifications for constructing reservoirs."""

from dataclasses import asdict, dataclass, field, replace
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

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | "ReadoutSpec" | None) -> "ReadoutSpec":
        """Build a readout spec from a plain mapping."""

        if data is None:
            return cls()
        if isinstance(data, cls):
            return data
        raw = dict(data)
        if "custom" in raw and raw["custom"] is not None:
            raw["custom"] = tuple(str(item) for item in raw["custom"])
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-safe dictionary."""

        out = asdict(self)
        out["custom"] = list(self.custom)
        if not isinstance(self.observables, str):
            out["observables"] = list(self.observables)
        return out


@dataclass(frozen=True)
class ReservoirSpec:
    """Task-agnostic reservoir construction spec.

    The construction source is intentionally separate from named presets. Built
    in reservoirs such as Ising and RandomPauli are conveniences that fill out
    this spec, while user-supplied Hamiltonians, circuits, or already-built
    reservoir objects can be attached directly.
    """

    family: str = "ising"
    preset: str | None = None
    source_kind: str = "preset"
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
    circuit_kwargs: Mapping[str, Any] = field(default_factory=dict)
    runtime: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def with_updates(self, **updates: Any) -> "ReservoirSpec":
        """Return a copy with selected fields replaced."""

        return replace(self, **updates)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | "ReservoirSpec") -> "ReservoirSpec":
        """Build a reservoir spec from a plain mapping."""

        if isinstance(data, cls):
            return data
        raw = dict(data)
        raw["readout"] = ReadoutSpec.from_mapping(raw.get("readout"))
        raw["model_kwargs"] = dict(raw.get("model_kwargs", {}))
        raw["hamiltonian_kwargs"] = dict(raw.get("hamiltonian_kwargs", {}))
        raw["circuit_kwargs"] = dict(raw.get("circuit_kwargs", {}))
        raw["runtime"] = dict(raw.get("runtime", {}))
        if raw.get("preset") is None:
            raw["preset"] = raw.get("family")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-safe dictionary."""

        out = {
            "family": self.family,
            "preset": self.preset,
            "source_kind": self.source_kind,
            "n_system": self.n_system,
            "n_ancilla": self.n_ancilla,
            "n_memory": self.n_memory,
            "n_readout": self.n_readout,
            "tau": self.tau,
            "input_scale": self.input_scale,
            "seed": self.seed,
            "readout": self.readout.to_dict(),
            "model_kwargs": dict(self.model_kwargs),
            "hamiltonian_kwargs": dict(self.hamiltonian_kwargs),
            "circuit_kwargs": dict(self.circuit_kwargs),
        }
        if self.runtime:
            out["runtime"] = {key: repr(value) for key, value in self.runtime.items()}
        return out

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
