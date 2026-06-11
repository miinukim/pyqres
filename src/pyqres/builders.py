from __future__ import annotations

"""High-level reservoir construction helpers."""

from typing import Any

import numpy as np

from pyqres.core import ReservoirParams

from .specs import ReadoutSpec, ReservoirSpec


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

    from pyqres.dim import IsingReservoirModel, IsingReservoirParameters

    family = spec.family.lower()
    if family != "ising":
        raise ValueError(f"Unsupported dimension model family '{spec.family}'")
    params = IsingReservoirParameters(
        n_memory=spec.system_qubits,
        n_readout=spec.ancilla_qubits,
        tau=float(spec.tau),
        **dict(spec.model_kwargs),
    )
    return IsingReservoirModel(params)


def build_hamiltonian_params(spec: ReservoirSpec) -> dict[str, Any]:
    """Build backend-neutral Hamiltonian parameters for simulation backends."""

    family = spec.family.lower()
    if family != "ising":
        raise ValueError(f"Unsupported Hamiltonian family '{spec.family}'")
    return ReservoirParams.ising_type(
        n_system=spec.system_qubits,
        n_ancilla=spec.ancilla_qubits,
        tau=float(spec.tau),
        seed=int(spec.seed),
        **dict(spec.hamiltonian_kwargs),
    ).generate()


def compile_reservoir(spec: ReservoirSpec, backend: str = "exact") -> Any:
    """Compile a ReservoirSpec into an executable reservoir."""

    backend_key = backend.lower()
    if backend_key in {"exact", "dense"} and spec.readout.mode in {"memory_observables", "observables"}:
        backend_key = "memory_observable"
    readout = spec.readout
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

        params = build_hamiltonian_params(spec)
        return QRCReservoir(
            QRCConfig(
                n_system=spec.system_qubits,
                n_ancilla=spec.ancilla_qubits,
                tau=float(spec.tau),
                input_scale=float(spec.input_scale),
                H0_hamiltonian=params["H0_hamiltonian"],
                H1_hamiltonian=params["H1_hamiltonian"],
                seed=int(spec.seed),
                reservoir_type="pauli_evolution",
            )
        )
    raise ValueError(f"Unsupported backend '{backend}'")


def transform(reservoir: Any, inputs: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    """Run any pyqres-compatible reservoir on a stream."""

    if hasattr(reservoir, "transform"):
        return np.asarray(reservoir.transform(inputs), dtype=float)
    if hasattr(reservoir, "run_stream"):
        return np.asarray(reservoir.run_stream(inputs), dtype=float)
    if hasattr(reservoir, "run"):
        return np.asarray(reservoir.run(inputs), dtype=float)
    raise TypeError("reservoir must expose transform, run_stream, or run")
