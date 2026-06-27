from __future__ import annotations

"""Preset adapters for common reservoir specs and model families.

The core package is intentionally generic: it knows how to compile explicit
Hamiltonians, circuits, and existing reservoir objects. Named models such as
Ising, RandomPauli, and SYK live here as adapters that fill in generic
``ReservoirSpec`` fields or construct the dimension-analysis model requested by
memory-observable readout.
"""

from typing import Any

from pyqres.core.reservoir_params import ReservoirParams
from pyqres.core.specs import ReadoutSpec, ReservoirSpec


def ising_memory_readout(
    *,
    n_memory: int = 4,
    n_readout: int = 1,
    tau: float = 1.0,
    observables: str = "z",
    observable_count: int | None = None,
    seed: int = 17462,
) -> ReservoirSpec:
    """Return a basic Ising memory/readout reservoir spec."""

    return ReservoirSpec(
        family="ising",
        preset="ising",
        source_kind="preset",
        n_memory=n_memory,
        n_readout=n_readout,
        tau=tau,
        seed=seed,
        readout=ReadoutSpec(mode="memory_observables", observables=observables, count=observable_count, init_state="zero"),
    )


def random_pauli_memory_readout(
    *,
    n_memory: int = 5,
    n_readout: int = 1,
    depth: int = 3,
    seed: int = 1234,
    observables: str = "z",
    observable_count: int | None = None,
) -> ReservoirSpec:
    """Return a RandomPauli dimension-model reservoir spec."""

    return ReservoirSpec(
        family="random_pauli",
        preset="random_pauli",
        source_kind="preset",
        n_memory=n_memory,
        n_readout=n_readout,
        seed=seed,
        model_kwargs={"depth": depth},
        readout=ReadoutSpec(mode="memory_observables", observables=observables, count=observable_count, init_state="zero"),
    )


def syk_memory_readout(
    *,
    n_memory: int = 7,
    n_readout: int = 1,
    tau: float = 1.0,
    seed: int = 1234,
    observables: str = "occupation",
    observable_count: int | None = None,
) -> ReservoirSpec:
    """Return an SYK dimension-model reservoir spec."""

    return ReservoirSpec(
        family="syk",
        preset="syk",
        source_kind="preset",
        n_memory=n_memory,
        n_readout=n_readout,
        tau=tau,
        seed=seed,
        readout=ReadoutSpec(mode="memory_observables", observables=observables, count=observable_count, init_state="zero"),
    )


def names() -> list[str]:
    """List available built-in preset names."""

    return [
        "ising",
        "ising.memory_readout",
        "random_pauli",
        "randompauli",
        "random_pauli.memory_readout",
        "syk",
        "syk.memory_readout",
    ]


def preset_key(spec_or_name: ReservoirSpec | str | None) -> str:
    """Return the normalized preset key used by adapter registries."""

    if isinstance(spec_or_name, ReservoirSpec):
        name = spec_or_name.dynamics.name or spec_or_name.preset or spec_or_name.family or "ising"
    else:
        name = spec_or_name or "ising"
    return str(name).lower()


def build_dimension_model(spec: ReservoirSpec) -> Any:
    """Build a dimension-analysis model for a named preset."""

    if spec.source_kind.lower() == "object" and "model" in spec.runtime:
        return spec.runtime["model"]

    from pyqres.dim import (
        IsingReservoirModel,
        IsingReservoirParameters,
        RandomPauliReservoirModel,
        RandomPauliReservoirParameters,
        SYKReservoirModel,
        SYKReservoirParameters,
    )

    registry = {
        "ising": (IsingReservoirParameters, IsingReservoirModel),
        "ising.memory_readout": (IsingReservoirParameters, IsingReservoirModel),
        "random_pauli": (RandomPauliReservoirParameters, RandomPauliReservoirModel),
        "randompauli": (RandomPauliReservoirParameters, RandomPauliReservoirModel),
        "random_pauli.memory_readout": (RandomPauliReservoirParameters, RandomPauliReservoirModel),
        "syk": (SYKReservoirParameters, SYKReservoirModel),
        "syk.memory_readout": (SYKReservoirParameters, SYKReservoirModel),
    }
    key = preset_key(spec)
    try:
        parameter_cls, model_cls = registry[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported dimension-model preset '{key}'. Available presets: {names()}") from exc

    kwargs = {
        "n_memory": spec.system_qubits,
        "n_readout": spec.ancilla_qubits,
        **dict(spec.dynamics.parameters),
        **dict(spec.model_kwargs),
    }
    kwargs.update(_dimension_encoding_kwargs(spec))
    if "tau" in getattr(parameter_cls, "__dataclass_fields__", {}):
        kwargs.setdefault("tau", float(spec.tau))
    if "seed" in getattr(parameter_cls, "__dataclass_fields__", {}):
        kwargs.setdefault("seed", int(spec.seed))
    params = parameter_cls(**kwargs)
    return model_cls(params)


def _dimension_encoding_kwargs(spec: ReservoirSpec) -> dict[str, Any]:
    """Translate generic encoding metadata into dimension-model parameters."""

    encoding = spec.encoding
    if encoding.mode.lower() != "hamiltonian" or not encoding.operator:
        return {}
    out: dict[str, Any] = {
        "input_axis": str(encoding.operator).upper(),
        "input_strength": float(encoding.scale),
    }
    if encoding.targets:
        out["input_site"] = int(encoding.targets[0])
        if len(encoding.targets) > 1:
            out["input_sites"] = tuple(int(item) for item in encoding.targets)
    params = dict(encoding.parameters)
    if "on_memory" in params:
        out["input_on_memory"] = bool(params["on_memory"])
    if "normalization" in params:
        out["input_strength_normalization"] = str(params["normalization"])
    return out


def _hamiltonian_encoding_kwargs(spec: ReservoirSpec) -> dict[str, Any]:
    """Translate generic encoding metadata into Hamiltonian preset parameters."""

    encoding = spec.encoding
    if encoding.mode.lower() != "hamiltonian" or str(encoding.operator or "").upper() != "Z":
        return {}
    return {
        "input_z_field_base": float(encoding.scale),
        "input_z_field_std": 0.0,
        "input_z_field_scale": 1.0,
    }


def build_hamiltonian_params(spec: ReservoirSpec) -> dict[str, Any]:
    """Build backend-neutral Hamiltonian parameters for Hamiltonian presets."""

    key = preset_key(spec)
    kwargs = {
        **dict(spec.dynamics.parameters),
        **dict(spec.hamiltonian_kwargs),
    }
    kwargs.update(_hamiltonian_encoding_kwargs(spec))
    for promoted in ("tau", "seed", "n_system", "n_ancilla", "n_memory", "n_readout", "input_scale"):
        kwargs.pop(promoted, None)
    if key in {"ising", "ising.memory_readout"}:
        return ReservoirParams.ising_type(
            n_system=spec.system_qubits,
            n_ancilla=spec.ancilla_qubits,
            tau=float(spec.tau),
            seed=int(spec.seed),
            **kwargs,
        ).generate()
    raise ValueError(
        f"Preset '{key}' does not define a dense Hamiltonian. "
        "Provide explicit Hamiltonian dynamics for exact/qiskit Hamiltonian evolution, "
        "or use backend='memory_observable' for dimension-model presets."
    )


def build_qiskit_artifacts(spec: ReservoirSpec) -> dict[str, Any]:
    """Build Qiskit-native artifacts required by the Qiskit reservoir backend.

    Compatibility wrapper for the core Hamiltonian-to-Qiskit adapter. Presets
    are only one possible source of Hamiltonians; explicit user Hamiltonians use
    the same conversion path.
    """

    from pyqres.core.builders import build_qiskit_hamiltonian_artifacts

    return build_qiskit_hamiltonian_artifacts(spec)


def get(name: str, **kwargs: object) -> ReservoirSpec:
    """Instantiate a named preset."""

    key = name.lower()
    if key in {"ising", "ising.memory_readout"}:
        return ising_memory_readout(**kwargs)
    if key in {"random_pauli", "randompauli", "random_pauli.memory_readout"}:
        return random_pauli_memory_readout(**kwargs)
    if key in {"syk", "syk.memory_readout"}:
        return syk_memory_readout(**kwargs)
    raise ValueError(f"Unknown preset '{name}'. Available presets: {names()}")
