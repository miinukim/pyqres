from __future__ import annotations

"""High-level reservoir construction helpers."""

from typing import Any, Mapping

import numpy as np

from pyqres.core.reservoir_params import ReservoirParams
from pyqres.core.protocols import InputSequence

from pyqres.core.specs import ReadoutSpec, ReservoirSpec


def _select_observable_specs(model: Any, readout: ReadoutSpec) -> list[str]:
    observables = readout.observables
    if isinstance(observables, str):
        specs = model.default_memory_observable_specs(preset=observables, custom_specs=list(readout.custom))
    else:
        specs = list(observables) + list(readout.custom)
    if readout.count is not None:
        count = int(readout.count)
        if count < 1 or count > len(specs):
            raise ValueError(f"observable count {count} outside valid range [1, {len(specs)}]")
        specs = specs[:count]
    return list(dict.fromkeys(specs))


def _preset_key(spec: ReservoirSpec) -> str:
    return str(spec.preset or spec.family or "ising").lower()


def build_dimension_model(spec: ReservoirSpec) -> Any:
    """Build a dimension-analysis model from a reservoir spec."""

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

    preset = _preset_key(spec)
    model_registry = {
        "ising": (IsingReservoirParameters, IsingReservoirModel),
        "ising.memory_readout": (IsingReservoirParameters, IsingReservoirModel),
        "random_pauli": (RandomPauliReservoirParameters, RandomPauliReservoirModel),
        "randompauli": (RandomPauliReservoirParameters, RandomPauliReservoirModel),
        "syk": (SYKReservoirParameters, SYKReservoirModel),
    }
    try:
        parameter_cls, model_cls = model_registry[preset]
    except KeyError as exc:
        raise ValueError(f"Unsupported dimension-model preset '{preset}'") from exc

    kwargs = {
        "n_memory": spec.system_qubits,
        "n_readout": spec.ancilla_qubits,
        **dict(spec.model_kwargs),
    }
    if "tau" in getattr(parameter_cls, "__dataclass_fields__", {}):
        kwargs.setdefault("tau", float(spec.tau))
    if "seed" in getattr(parameter_cls, "__dataclass_fields__", {}):
        kwargs.setdefault("seed", int(spec.seed))
    params = parameter_cls(**kwargs)
    return model_cls(params)


def _normalize_hamiltonian_kwargs(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize ergonomic public Hamiltonian aliases to ReservoirParams keys."""

    kwargs = dict(raw)
    aliases = {
        "h0": "h0_matrix",
        "h1": "h1_matrix",
        "H0": "h0_matrix",
        "H1": "h1_matrix",
        "H0_matrix": "h0_matrix",
        "H1_matrix": "h1_matrix",
        "H0_hamiltonian": "h0_hamiltonian",
        "H1_hamiltonian": "h1_hamiltonian",
    }
    for source, target in aliases.items():
        if source in kwargs:
            kwargs[target] = kwargs.pop(source)
    if "kind" in kwargs and "hamiltonian_kind" not in kwargs:
        kwargs["hamiltonian_kind"] = kwargs.pop("kind")
    return kwargs


def build_hamiltonian_params(spec: ReservoirSpec) -> dict[str, Any]:
    """Build backend-neutral Hamiltonian parameters for simulation backends."""

    kwargs = _normalize_hamiltonian_kwargs(spec.hamiltonian_kwargs)
    source_kind = spec.source_kind.lower()
    if source_kind == "hamiltonian":
        kwargs.setdefault("n_system", spec.system_qubits)
        kwargs.setdefault("n_ancilla", spec.ancilla_qubits)
        kwargs.setdefault("tau", float(spec.tau))
        kwargs.setdefault("seed", int(spec.seed))
        if "hamiltonian_kind" not in kwargs:
            if "h0_terms" in kwargs or "h1_terms" in kwargs:
                kwargs["hamiltonian_kind"] = "pauli_terms"
            else:
                kwargs["hamiltonian_kind"] = "matrix"
        return ReservoirParams(**kwargs).generate()

    preset = _preset_key(spec)
    if preset in {"ising", "ising.memory_readout"}:
        return ReservoirParams.ising_type(
            n_system=spec.system_qubits,
            n_ancilla=spec.ancilla_qubits,
            tau=float(spec.tau),
            seed=int(spec.seed),
            **kwargs,
        ).generate()
    raise ValueError(
        f"Preset '{preset}' does not define a dense Hamiltonian. "
        "Provide .hamiltonian(...) for exact/qiskit Hamiltonian evolution, "
        "or use backend='memory_observable' for dimension-model presets."
    )


def compile_reservoir(spec: ReservoirSpec, backend: str = "exact") -> Any:
    """Compile a ReservoirSpec into an executable reservoir."""

    if spec.source_kind.lower() == "object":
        if "reservoir" not in spec.runtime:
            raise ValueError("source_kind='object' requires runtime['reservoir'].")
        return spec.runtime["reservoir"]

    backend_key = backend.lower()
    if backend_key in {"exact", "dense"} and spec.readout.mode in {"memory_observables", "observables"}:
        backend_key = "memory_observable"
    readout = spec.readout
    if spec.source_kind.lower() == "circuit" and backend_key != "qiskit":
        raise ValueError("Custom circuit reservoirs currently compile with backend='qiskit'.")
    if backend_key in {"exact", "channel_map"}:
        from pyqres.simulation import ChannelMapReservoir, ChannelMapReservoirConfig

        params = build_hamiltonian_params(spec)
        return ChannelMapReservoir(
            ChannelMapReservoirConfig(
                n_system=spec.system_qubits,
                n_ancilla=spec.ancilla_qubits,
                tau=float(spec.tau),
                input_scale=float(spec.input_scale),
                include_bias=bool(readout.include_bias),
                use_shot_noise=bool(readout.use_shot_noise),
                shots=int(readout.shots),
                init_state=str(readout.init_state),
                H0_hamiltonian=params["H0_hamiltonian"],
                H1_hamiltonian=params["H1_hamiltonian"],
                seed=int(spec.seed),
            )
        )
    if backend_key in {"hardware", "hardware_trajectory"}:
        from pyqres.simulation import HardwareTrajectoryReservoir, HardwareTrajectoryReservoirConfig

        params = build_hamiltonian_params(spec)
        return HardwareTrajectoryReservoir(
            HardwareTrajectoryReservoirConfig(
                n_system=spec.system_qubits,
                n_ancilla=spec.ancilla_qubits,
                tau=float(spec.tau),
                input_scale=float(spec.input_scale),
                include_bias=bool(readout.include_bias),
                init_state=str(readout.init_state),
                shots=int(readout.shots),
                H0_hamiltonian=params["H0_hamiltonian"],
                H1_hamiltonian=params["H1_hamiltonian"],
                seed=int(spec.seed),
            )
        )
    if backend_key in {"memory_observable", "dim"}:
        from pyqres.dim import MemoryObservableStreamingReservoir

        model = build_dimension_model(spec)
        specs = _select_observable_specs(model, readout)
        observables = [model.parse_memory_observable(item) for item in specs]
        return MemoryObservableStreamingReservoir(
            model=model,
            observables=observables,
            include_bias=bool(readout.include_bias),
            init_state=str(readout.init_state),
        )
    if backend_key == "qiskit":
        from pyqres.qiskit import QRCConfig, QRCReservoir

        circuit_kwargs = dict(spec.circuit_kwargs)
        if spec.source_kind.lower() == "circuit":
            if "circuit" not in spec.runtime:
                raise ValueError("Circuit reservoirs require a runtime circuit object.")
            circuit_kwargs.pop("reservoir_type", None)
            circuit_kwargs.pop("reservoir_circuit", None)
            qiskit_kwargs = {
                "n_system": spec.system_qubits,
                "n_ancilla": spec.ancilla_qubits,
                "tau": float(spec.tau),
                "input_scale": float(spec.input_scale),
                "seed": int(spec.seed),
                "include_bias": bool(readout.include_bias),
                "shots": int(readout.shots),
                "reservoir_type": "custom_circuit",
                "reservoir_circuit": spec.runtime["circuit"],
            }
            qiskit_kwargs.update(circuit_kwargs)
            return QRCReservoir(
                QRCConfig(**qiskit_kwargs)
            )

        params = build_hamiltonian_params(spec)
        qiskit_kwargs = {
            "n_system": spec.system_qubits,
            "n_ancilla": spec.ancilla_qubits,
            "tau": float(spec.tau),
            "input_scale": float(spec.input_scale),
            "H0_hamiltonian": params["H0_hamiltonian"],
            "H1_hamiltonian": params["H1_hamiltonian"],
            "seed": int(spec.seed),
            "include_bias": bool(readout.include_bias),
            "shots": int(readout.shots),
            "reservoir_type": "pauli_evolution",
        }
        qiskit_kwargs.update(circuit_kwargs)
        return QRCReservoir(
            QRCConfig(**qiskit_kwargs)
        )
    raise ValueError(f"Unsupported backend '{backend}'")


def transform(reservoir: Any, inputs: InputSequence) -> np.ndarray:
    """Run any pyqres-compatible reservoir on a stream."""

    if hasattr(reservoir, "transform"):
        return np.asarray(reservoir.transform(inputs), dtype=float)
    if hasattr(reservoir, "run_stream"):
        return np.asarray(reservoir.run_stream(inputs), dtype=float)
    if hasattr(reservoir, "run"):
        return np.asarray(reservoir.run(inputs), dtype=float)
    raise TypeError("reservoir must expose transform, run_stream, or run")
