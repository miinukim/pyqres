from __future__ import annotations

"""High-level reservoir construction helpers."""

from dataclasses import fields
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


def build_dimension_model(spec: ReservoirSpec) -> Any:
    """Build a dimension-analysis model from a reservoir spec."""

    if spec.source_kind.lower() == "object" and "model" in spec.runtime:
        return spec.runtime["model"]

    from pyqres import presets

    return presets.build_dimension_model(spec)


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

    kwargs = _normalize_hamiltonian_kwargs({**dict(spec.dynamics.parameters), **dict(spec.hamiltonian_kwargs)})
    source_kind = spec.source_kind.lower()
    dynamics_kind = spec.dynamics.kind.lower()
    if source_kind == "hamiltonian" or dynamics_kind == "hamiltonian":
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

    from pyqres import presets

    return presets.build_hamiltonian_params(spec)


def build_qiskit_hamiltonian_artifacts(spec: ReservoirSpec) -> dict[str, Any]:
    """Build Qiskit-native Hamiltonian artifacts for PauliEvolutionGate dynamics.

    This is the Qiskit-side equivalent of build_hamiltonian_params: explicit
    user Hamiltonians and preset-generated Hamiltonians are both normalized to
    SparsePauliOp artifacts before entering the low-level Qiskit reservoir.
    """

    try:
        from qiskit.quantum_info import Operator, SparsePauliOp
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("qiskit is required to build Qiskit Hamiltonian artifacts.") from exc

    from pyqres.core.reservoir_params import dense_hamiltonian_matrix

    n_qubits = spec.system_qubits + spec.ancilla_qubits

    def to_sparse_pauli_op(value: Any) -> Any:
        if value is None:
            return SparsePauliOp.from_list([("I" * n_qubits, 0.0)])
        if isinstance(value, SparsePauliOp):
            return value
        if hasattr(value, "to_sparse_pauli_op") and callable(value.to_sparse_pauli_op):
            return value.to_sparse_pauli_op()
        return SparsePauliOp.from_operator(Operator(dense_hamiltonian_matrix(value)))

    params = build_hamiltonian_params(spec)
    return {
        "reservoir_type": "pauli_evolution",
        "H0_hamiltonian": to_sparse_pauli_op(params["H0_hamiltonian"]),
        "H1_hamiltonian": to_sparse_pauli_op(params["H1_hamiltonian"]),
    }


_PRETHERMAL_PRESETS = {"prethermal_shadow", "global_floquet_shadow", "global_floquet.partial_shadow"}


def _preset_key(spec: ReservoirSpec) -> str:
    return str(spec.dynamics.name or spec.preset or spec.family or "").lower()


def _dataclass_kwargs(cls: type, values: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {field.name for field in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return {key: value for key, value in values.items() if key in allowed}


def _build_prethermal_shadow_reservoir(spec: ReservoirSpec, backend: str) -> Any:
    from pyqres.prethermal_shadow import (
        GlobalFloquetConfig,
        GlobalFloquetPartialShadowReservoir,
        InputEncodingConfig,
        PartialShadowReadoutConfig,
        PrethermalCircuitConfig,
        QiskitGlobalFloquetPartialShadowReservoir,
        ReadoutResetConfig,
    )

    model_kwargs = dict(spec.model_kwargs)
    floquet = dict(model_kwargs.pop("floquet", {}))
    shadow = dict(model_kwargs.pop("shadow", {}))
    reset = dict(model_kwargs.pop("reset", {}))
    simulator = dict(model_kwargs.pop("simulator", {}))
    circuit = dict(model_kwargs.pop("circuit", {}))
    if model_kwargs:
        floquet.update(model_kwargs)

    floquet.setdefault("n_memory", spec.system_qubits)
    floquet.setdefault("n_readout", spec.ancilla_qubits)
    floquet.setdefault("n_qubits", spec.system_qubits + spec.ancilla_qubits)
    floquet.setdefault("seed", int(spec.seed))
    if "omega" not in floquet:
        raise ValueError("prethermal_shadow preset requires floquet.omega.")
    if "n_cycles_per_step" not in floquet:
        raise ValueError("prethermal_shadow preset requires floquet.n_cycles_per_step.")

    encoding = dict(spec.encoding.parameters)
    if spec.encoding.targets:
        encoding.setdefault("input_qubits", tuple(spec.encoding.targets))
    elif "input_qubits" not in encoding:
        encoding.setdefault("input_qubits", "memory")
    if spec.encoding.operator is not None:
        encoding.setdefault("axis", str(spec.encoding.operator).lower())
    encoding.setdefault("beta", float(spec.encoding.scale))
    encoding.setdefault("bias", float(spec.encoding.bias))

    shadow.setdefault("include_bias", bool(spec.readout.include_bias))
    shadow.setdefault("shots", int(spec.readout.shots))

    configs = (
        GlobalFloquetConfig(**_dataclass_kwargs(GlobalFloquetConfig, floquet)),
        InputEncodingConfig(**_dataclass_kwargs(InputEncodingConfig, encoding)),
        PartialShadowReadoutConfig(**_dataclass_kwargs(PartialShadowReadoutConfig, shadow)),
        ReadoutResetConfig(**_dataclass_kwargs(ReadoutResetConfig, reset)),
    )
    backend_key = str(backend).lower()
    if backend_key in {"qiskit", "circuit", "mps"}:
        circuit = {**circuit, **dict(spec.qiskit_kwargs), **simulator}
        return QiskitGlobalFloquetPartialShadowReservoir(
            *configs,
            PrethermalCircuitConfig(**_dataclass_kwargs(PrethermalCircuitConfig, circuit)),
        )
    return GlobalFloquetPartialShadowReservoir(*configs, **simulator)


def _hamiltonian_backend_kwargs(spec: ReservoirSpec) -> dict[str, Any]:
    readout = spec.readout
    params = build_hamiltonian_params(spec)
    return {
        "n_system": spec.system_qubits,
        "n_ancilla": spec.ancilla_qubits,
        "tau": float(spec.tau),
        "input_scale": float(spec.input_scale),
        "include_bias": bool(readout.include_bias),
        "init_state": str(readout.init_state),
        "shots": int(readout.shots),
        "H0_hamiltonian": params["H0_hamiltonian"],
        "H1_hamiltonian": params["H1_hamiltonian"],
        "seed": int(spec.seed),
    }


def _qiskit_base_kwargs(spec: ReservoirSpec) -> dict[str, Any]:
    readout = spec.readout
    return {
        "n_system": spec.system_qubits,
        "n_ancilla": spec.ancilla_qubits,
        "tau": float(spec.tau),
        "input_scale": float(spec.input_scale),
        "seed": int(spec.seed),
        "include_bias": bool(readout.include_bias),
        "shots": int(readout.shots),
    }


def _build_qiskit_reservoir(spec: ReservoirSpec) -> Any:
    from pyqres.qiskit import QRCConfig, QRCReservoir

    circuit_kwargs = dict(spec.circuit_kwargs)
    qiskit_kwargs = {**_qiskit_base_kwargs(spec), **dict(spec.qiskit_kwargs)}
    if spec.source_kind.lower() == "circuit":
        if "circuit" not in spec.runtime:
            raise ValueError("Circuit reservoirs require a runtime circuit object.")
        circuit_kwargs.pop("reservoir_type", None)
        circuit_kwargs.pop("reservoir_circuit", None)
        qiskit_kwargs.update(
            {
                "reservoir_type": "custom_circuit",
                "reservoir_circuit": spec.runtime["circuit"],
            }
        )
    else:
        qiskit_kwargs.update(build_qiskit_hamiltonian_artifacts(spec))

    qiskit_kwargs.update(circuit_kwargs)
    return QRCReservoir(QRCConfig(**qiskit_kwargs))


def compile_reservoir(spec: ReservoirSpec, backend: str = "exact") -> Any:
    """Compile a ReservoirSpec into an executable reservoir."""

    if spec.source_kind.lower() == "object":
        if "reservoir" not in spec.runtime:
            raise ValueError("source_kind='object' requires runtime['reservoir'].")
        return spec.runtime["reservoir"]

    backend_key = backend.lower()
    if spec.source_kind.lower() == "preset" and _preset_key(spec) in _PRETHERMAL_PRESETS:
        return _build_prethermal_shadow_reservoir(spec, backend_key)
    if backend_key in {"exact", "dense"} and spec.readout.mode in {"memory_observables", "observables"}:
        backend_key = "memory_observable"
    readout = spec.readout
    if spec.source_kind.lower() == "circuit" and backend_key != "qiskit":
        raise ValueError("Custom circuit reservoirs currently compile with backend='qiskit'.")
    if backend_key in {"exact", "channel_map"}:
        from pyqres.simulation import ChannelMapReservoir, ChannelMapReservoirConfig

        kwargs = _hamiltonian_backend_kwargs(spec)
        return ChannelMapReservoir(
            ChannelMapReservoirConfig(
                use_shot_noise=bool(readout.use_shot_noise),
                **kwargs,
            )
        )
    if backend_key in {"hardware", "hardware_trajectory"}:
        from pyqres.simulation import HardwareTrajectoryReservoir, HardwareTrajectoryReservoirConfig

        return HardwareTrajectoryReservoir(HardwareTrajectoryReservoirConfig(**_hamiltonian_backend_kwargs(spec)))
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
        return _build_qiskit_reservoir(spec)
    raise ValueError(f"Unsupported backend '{backend}'")


def run(reservoir: Any, inputs: InputSequence) -> np.ndarray:
    """Run any pyqres-compatible reservoir on a stream."""

    for method_name in ("run", "run_stream", "transform"):
        method = getattr(reservoir, method_name, None)
        if method is not None:
            return np.asarray(method(inputs), dtype=float)
    raise TypeError("reservoir must expose transform, run_stream, or run")
