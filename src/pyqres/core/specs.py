from __future__ import annotations

"""Serializable public specifications for constructing reservoirs."""

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class InputEncodingSpec:
    """Task-agnostic description of how scalar/vector inputs enter a reservoir.

    The core does not interpret every possible encoding. It preserves a compact
    representation that compilers, presets, or user-provided reservoirs can
    interpret. Built-in adapters currently understand Hamiltonian modulation
    and circuit-level encoding options, while custom code can attach arbitrary
    parameters through ``parameters``.
    """

    mode: str = "hamiltonian"
    operator: str | None = None
    targets: tuple[int, ...] = ()
    scale: float = 1.0
    bias: float = 0.0
    parameters: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | "InputEncodingSpec" | None) -> "InputEncodingSpec":
        """Build an input-encoding spec from a plain mapping."""

        if data is None:
            return cls()
        if isinstance(data, cls):
            return data
        raw = dict(data)
        if "axis" in raw and "operator" not in raw:
            raw["operator"] = raw.pop("axis")
        if "site" in raw and "targets" not in raw:
            raw["targets"] = (raw.pop("site"),)
        if "sites" in raw and "targets" not in raw:
            raw["targets"] = raw.pop("sites")
        if "strength" in raw and "scale" not in raw:
            raw["scale"] = raw.pop("strength")
        if raw.get("operator") is not None:
            raw["operator"] = str(raw["operator"]).upper()
        parameters = dict(raw.get("parameters", {}))
        known = {"mode", "operator", "targets", "scale", "bias", "parameters"}
        for key in list(raw):
            if key not in known:
                parameters[key] = raw.pop(key)
        raw["targets"] = tuple(int(item) for item in raw.get("targets", ()))
        raw["scale"] = float(raw.get("scale", 1.0))
        raw["bias"] = float(raw.get("bias", 0.0))
        raw["parameters"] = parameters
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-safe dictionary."""

        return {
            "mode": self.mode,
            "operator": self.operator,
            "targets": list(self.targets),
            "scale": self.scale,
            "bias": self.bias,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class DynamicsSpec:
    """Task-agnostic description of reservoir dynamics.

    Dynamics may be a named preset, explicit Hamiltonian, user circuit, existing
    reservoir object, or another compiler-specific scheme. Presets are resolved
    outside core in ``pyqres.presets``; core compilers consume explicit dynamics
    fields and delegate named presets.
    """

    kind: str = "preset"
    name: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | "DynamicsSpec" | None) -> "DynamicsSpec":
        """Build a dynamics spec from a plain mapping."""

        if data is None:
            return cls()
        if isinstance(data, cls):
            return data
        raw = dict(data)
        if "preset" in raw and "name" not in raw:
            raw["name"] = raw.pop("preset")
            raw.setdefault("kind", "preset")
        if "family" in raw and "name" not in raw:
            raw["name"] = raw.pop("family")
            raw.setdefault("kind", "preset")
        parameters = dict(raw.get("parameters", {}))
        known = {"kind", "name", "parameters"}
        for key in list(raw):
            if key not in known:
                parameters[key] = raw.pop(key)
        raw["parameters"] = parameters
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-safe dictionary."""

        return {
            "kind": self.kind,
            "name": self.name,
            "parameters": dict(self.parameters),
        }


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
    encoding: InputEncodingSpec = field(default_factory=InputEncodingSpec)
    dynamics: DynamicsSpec = field(default_factory=DynamicsSpec)
    readout: ReadoutSpec = field(default_factory=ReadoutSpec)
    model_kwargs: Mapping[str, Any] = field(default_factory=dict)
    hamiltonian_kwargs: Mapping[str, Any] = field(default_factory=dict)
    circuit_kwargs: Mapping[str, Any] = field(default_factory=dict)
    qiskit_kwargs: Mapping[str, Any] = field(default_factory=dict)
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
        raw["encoding"] = InputEncodingSpec.from_mapping(raw.get("encoding"))
        raw["dynamics"] = DynamicsSpec.from_mapping(raw.get("dynamics"))
        raw["readout"] = ReadoutSpec.from_mapping(raw.get("readout"))
        raw["model_kwargs"] = dict(raw.get("model_kwargs", {}))
        raw["hamiltonian_kwargs"] = dict(raw.get("hamiltonian_kwargs", {}))
        raw["circuit_kwargs"] = dict(raw.get("circuit_kwargs", {}))
        raw["qiskit_kwargs"] = dict(raw.get("qiskit_kwargs", {}))
        raw["runtime"] = dict(raw.get("runtime", {}))
        if raw.get("preset") is None:
            raw["preset"] = raw.get("family")
        if raw["dynamics"].name is None:
            raw["dynamics"] = replace(raw["dynamics"], name=raw.get("preset") or raw.get("family"))
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
            "encoding": self.encoding.to_dict(),
            "dynamics": self.dynamics.to_dict(),
            "readout": self.readout.to_dict(),
            "model_kwargs": dict(self.model_kwargs),
            "hamiltonian_kwargs": dict(self.hamiltonian_kwargs),
            "circuit_kwargs": dict(self.circuit_kwargs),
            "qiskit_kwargs": dict(self.qiskit_kwargs),
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
