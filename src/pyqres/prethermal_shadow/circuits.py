from __future__ import annotations

"""Qiskit circuit backend for the global-Floquet partial-shadow reservoir."""

from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np

from .config import (
    GlobalFloquetConfig,
    InputEncodingConfig,
    PartialShadowReadoutConfig,
    PrethermalCircuitConfig,
    ReadoutResetConfig,
)
from .dynamics import (
    PauliTerm,
    drive_pauli_terms,
    fast_period,
    generate_hamiltonian_parameters,
    global_h0_pauli_terms,
    input_beta_array,
    parse_pauli_terms,
    resolve_input_qubits,
    resolve_qubit_partition,
    step_duration,
    validate_axis,
    validate_floquet_config,
)
from .shadows import (
    counts_to_outcomes,
    estimate_shadow_features,
    generate_pauli_labels,
    partial_shadow_feature_names,
    sample_basis_schedules,
)

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.synthesis import LieTrotter, SuzukiTrotter
except Exception:  # pragma: no cover - optional dependency
    QuantumCircuit = None  # type: ignore[assignment]
    transpile = None  # type: ignore[assignment]
    PauliEvolutionGate = None  # type: ignore[assignment]
    SparsePauliOp = None  # type: ignore[assignment]
    LieTrotter = None  # type: ignore[assignment]
    SuzukiTrotter = None  # type: ignore[assignment]

try:
    from qiskit_aer import AerSimulator
except Exception:  # pragma: no cover - optional dependency
    AerSimulator = None  # type: ignore[assignment]


def _require_qiskit() -> None:
    if QuantumCircuit is None or SparsePauliOp is None or PauliEvolutionGate is None:
        raise ImportError(
            "The prethermal circuit backend requires qiskit (pip install 'pyqres[qiskit]')."
        )


def _qiskit_pauli_label(n_qubits: int, placements: Sequence[tuple[int, str]]) -> str:
    """Map physical qubit indices to Qiskit's little-endian Pauli labels."""

    label = ["I"] * int(n_qubits)
    for qubit, pauli in placements:
        index = int(qubit)
        if index < 0 or index >= int(n_qubits):
            raise ValueError(
                f"Pauli placement qubit {index} is outside [0, {int(n_qubits) - 1}]."
            )
        label[int(n_qubits) - index - 1] = str(pauli).upper()
    return "".join(label)


def _sparse_pauli_op(n_qubits: int, terms: Sequence[PauliTerm]) -> Any:
    _require_qiskit()
    items = [
        (_qiskit_pauli_label(n_qubits, placements), complex(coeff))
        for coeff, placements in terms
    ]
    if not items:
        items = [("I" * int(n_qubits), 0.0)]
    return SparsePauliOp.from_list(items).simplify()


class QiskitGlobalFloquetPartialShadowReservoir:
    """Streaming prethermal QRC executed as dynamic Qiskit circuits.

    Each shadow shot is a complete temporal trajectory. At every input step the
    circuit applies input encoding and Floquet evolution, measures the selected
    readout qubits in scheduled local Pauli bases, resets them, and continues
    evolving the unmeasured memory qubits. This is directly compatible with Aer
    MPS and with hardware backends that support mid-circuit measurement/reset.
    """

    def __init__(
        self,
        floquet_config: GlobalFloquetConfig,
        input_config: InputEncodingConfig | None = None,
        shadow_config: PartialShadowReadoutConfig | None = None,
        reset_config: ReadoutResetConfig | None = None,
        circuit_config: PrethermalCircuitConfig | None = None,
        backend: Any | None = None,
        seed_simulator: int | None = None,
    ):
        _require_qiskit()
        validate_floquet_config(floquet_config)
        self.floquet_config = floquet_config
        self.input_config = input_config or InputEncodingConfig()
        self.shadow_config = shadow_config or PartialShadowReadoutConfig()
        self.reset_config = reset_config or ReadoutResetConfig()
        self.circuit_config = circuit_config or PrethermalCircuitConfig()
        self.backend = backend

        self.n_qubits = int(floquet_config.n_qubits)
        self.n_memory = int(floquet_config.n_memory)
        self.n_readout = int(floquet_config.n_readout)
        self.memory_qubits, self.readout_qubits = resolve_qubit_partition(
            floquet_config
        )
        self.delta_t = step_duration(floquet_config)
        self._validate_configs()

        params = generate_hamiltonian_parameters(floquet_config)
        self.edges = params["edges"]
        self.h = np.asarray(params["h"], dtype=float)
        self.jz = np.asarray(params["jz"], dtype=float)
        self.jxy = np.asarray(params["jxy"], dtype=float)
        self.drive_coeffs = np.asarray(params["drive_coeffs"], dtype=float)
        self.break_coeffs = np.asarray(params["break_coeffs"], dtype=float)
        h0_terms = global_h0_pauli_terms(
            floquet_config,
            h=self.h,
            jz=self.jz,
            jxy=self.jxy,
            break_coeffs=self.break_coeffs,
            edges=self.edges,
        )
        self.h0_hamiltonian = _sparse_pauli_op(self.n_qubits, h0_terms)
        self.drive_hamiltonian = _sparse_pauli_op(
            self.n_qubits,
            drive_pauli_terms(floquet_config, self.drive_coeffs),
        )

        self.input_operator_hamiltonian: Any | None = None
        if self.input_config.operator is None:
            self.input_qubits = resolve_input_qubits(self.input_config, floquet_config)
            self.input_axis = validate_axis(self.input_config.axis, "input axis")
            self.beta = input_beta_array(self.input_config, len(self.input_qubits))
        else:
            if not isinstance(self.input_config.beta, (int, float, np.floating)):
                raise ValueError("operator input encoding requires scalar beta.")
            self.input_qubits = tuple()
            self.input_axis = validate_axis(self.input_config.axis, "input axis")
            self.beta = np.asarray([float(self.input_config.beta)], dtype=float)
            operator_terms = parse_pauli_terms(
                str(self.input_config.operator),
                normalize=bool(self.input_config.normalize_operator),
            )
            if operator_terms:
                self.input_operator_hamiltonian = _sparse_pauli_op(
                    self.n_qubits, operator_terms
                )

        self.pauli_labels = generate_pauli_labels(
            self.n_readout, int(self.shadow_config.pauli_k)
        )
        self.feature_names = partial_shadow_feature_names(
            self.n_readout,
            int(self.shadow_config.pauli_k),
            include_bias=bool(self.shadow_config.include_bias),
        )
        configured_seed = (
            self.circuit_config.seed_simulator
            if seed_simulator is None
            else seed_simulator
        )
        self._rng_seed = int(
            self.shadow_config.seed if configured_seed is None else configured_seed
        )
        self.last_basis_schedules: np.ndarray | None = None
        self.last_outcomes: np.ndarray | None = None

    def _validate_configs(self) -> None:
        shadow = self.shadow_config
        circuit = self.circuit_config
        if str(shadow.feature_scope).lower() != "readout":
            raise ValueError(
                "The circuit backend supports feature_scope='readout' only."
            )
        if shadow.exact_expectations or not shadow.return_shadow_estimates:
            raise ValueError(
                "The circuit backend requires finite-shot shadow estimates."
            )
        if str(shadow.measurement_type).lower() != "projective":
            raise ValueError(
                "The circuit backend currently supports projective measurements only."
            )
        if shadow.basis_randomization != "local_pauli":
            raise ValueError(
                "The circuit backend requires basis_randomization='local_pauli'."
            )
        if int(shadow.shots) <= 0:
            raise ValueError("shadow shots must be positive.")
        if int(shadow.pauli_k) < 1 or int(shadow.pauli_k) > self.n_readout:
            raise ValueError(f"shadow pauli_k must lie in [1, {self.n_readout}].")
        if (
            not self.reset_config.reset_after_measurement
            or not self.reset_config.trace_readout_after_step
        ):
            raise NotImplementedError(
                "The circuit backend requires readout measurement and reset after every step."
            )
        if self.reset_config.reset_state not in {"zero", "plus"}:
            raise ValueError("reset_state must be zero or plus.")
        if circuit.evolution_synthesis not in {
            "default",
            "lie_trotter",
            "suzuki_trotter",
        }:
            raise ValueError(
                "evolution_synthesis must be default, lie_trotter, or suzuki_trotter."
            )
        if int(circuit.evolution_reps) <= 0:
            raise ValueError("evolution_reps must be positive.")
        if int(circuit.circuit_batch_size) <= 0:
            raise ValueError("circuit_batch_size must be positive.")
        if int(circuit.shots_per_basis) <= 0:
            raise ValueError("shots_per_basis must be positive.")
        if (
            circuit.mps_max_bond_dimension is not None
            and int(circuit.mps_max_bond_dimension) <= 0
        ):
            raise ValueError("mps_max_bond_dimension must be positive when provided.")
        if (
            circuit.mps_truncation_threshold is not None
            and float(circuit.mps_truncation_threshold) < 0.0
        ):
            raise ValueError(
                "mps_truncation_threshold must be non-negative when provided."
            )

    def _evolution_synthesis(self) -> Any | None:
        cfg = self.circuit_config
        if cfg.evolution_synthesis == "default":
            return None
        kwargs = {
            "reps": int(cfg.evolution_reps),
            "insert_barriers": bool(cfg.evolution_insert_barriers),
            "preserve_order": bool(cfg.evolution_preserve_order),
        }
        if cfg.evolution_synthesis == "lie_trotter":
            return LieTrotter(**kwargs)
        return SuzukiTrotter(order=int(cfg.evolution_order), **kwargs)

    def _append_input_encoding(self, circuit: Any, value: float) -> None:
        scale = float(value) + float(self.input_config.bias)
        if self.input_operator_hamiltonian is not None:
            circuit.append(
                PauliEvolutionGate(
                    self.input_operator_hamiltonian,
                    time=float(self.beta[0]) * scale,
                    synthesis=self._evolution_synthesis(),
                ),
                list(range(self.n_qubits)),
            )
            return
        for qubit, beta in zip(self.input_qubits, self.beta):
            angle = 2.0 * float(beta) * scale
            getattr(circuit, f"r{self.input_axis.lower()}")(angle, int(qubit))

    def _append_floquet_evolution(self, circuit: Any) -> None:
        half_period = 0.5 * fast_period(self.floquet_config)
        synthesis = self._evolution_synthesis()
        h_plus = self.h0_hamiltonian + self.drive_hamiltonian
        h_minus = self.h0_hamiltonian - self.drive_hamiltonian
        qubits = list(range(self.n_qubits))
        for _ in range(int(self.floquet_config.n_cycles_per_step)):
            circuit.append(
                PauliEvolutionGate(h_plus, time=half_period, synthesis=synthesis),
                qubits,
            )
            circuit.append(
                PauliEvolutionGate(h_minus, time=half_period, synthesis=synthesis),
                qubits,
            )

    def build_step_circuit(self, value: float) -> Any:
        """Build one unitary input-plus-Floquet step without measurement."""

        circuit = QuantumCircuit(self.n_qubits)
        self._append_input_encoding(circuit, float(value))
        self._append_floquet_evolution(circuit)
        return circuit

    def _prepare_readout_reset_state(self, circuit: Any) -> None:
        if self.reset_config.reset_state == "plus":
            for qubit in self.readout_qubits:
                circuit.h(int(qubit))

    @staticmethod
    def _append_basis_rotation(circuit: Any, qubit: int, basis: str) -> None:
        axis = str(basis).upper()
        if axis == "X":
            circuit.h(int(qubit))
        elif axis == "Y":
            circuit.sdg(int(qubit))
            circuit.h(int(qubit))
        elif axis != "Z":
            raise ValueError(f"unsupported measurement basis {basis!r}")

    def build_streaming_circuit(
        self,
        inputs: Sequence[float] | np.ndarray,
        basis_schedule: Sequence[Sequence[str]] | np.ndarray,
    ) -> Any:
        """Build one hardware-shaped trajectory circuit for a basis schedule."""

        values = np.asarray(inputs, dtype=float).reshape(-1)
        schedule = np.asarray(basis_schedule, dtype="U1")
        expected_shape = (values.size, self.n_readout)
        if schedule.shape != expected_shape:
            raise ValueError(
                f"basis_schedule must have shape {expected_shape}, got {schedule.shape}."
            )
        circuit = QuantumCircuit(self.n_qubits, values.size * self.n_readout)
        self._prepare_readout_reset_state(circuit)
        for step, value in enumerate(values):
            self._append_input_encoding(circuit, float(value))
            self._append_floquet_evolution(circuit)
            offset = step * self.n_readout
            for readout_index, qubit in enumerate(self.readout_qubits):
                self._append_basis_rotation(
                    circuit, int(qubit), str(schedule[step, readout_index])
                )
                circuit.measure(int(qubit), offset + readout_index)
            for qubit in self.readout_qubits:
                circuit.reset(int(qubit))
            if step + 1 < values.size:
                self._prepare_readout_reset_state(circuit)
        return circuit

    def build_executable_circuit(
        self,
        inputs: Sequence[float] | np.ndarray,
        basis_schedule: Sequence[Sequence[str]] | np.ndarray,
        backend: Any | None = None,
        **transpile_options: Any,
    ) -> Any:
        """Build and transpile one trajectory for Aer or a hardware backend."""

        if transpile is None:
            _require_qiskit()
        circuit = self.build_streaming_circuit(inputs, basis_schedule)
        target = self.backend if backend is None else backend
        if target is None:
            return circuit.decompose(reps=10)
        return transpile(
            circuit,
            backend=target,
            optimization_level=int(self.circuit_config.transpile_optimization_level),
            **transpile_options,
        )

    def _aer_backend(self) -> Any:
        if AerSimulator is None:
            raise ImportError(
                "Aer execution requires qiskit-aer (pip install 'pyqres[qiskit]')."
            )
        cfg = self.circuit_config
        options: dict[str, Any] = {
            "method": str(cfg.simulator_method),
            "seed_simulator": self._rng_seed,
            **dict(cfg.aer_options),
        }
        if str(cfg.simulator_device).lower() != "automatic":
            options["device"] = str(cfg.simulator_device).upper()
        if cfg.mps_max_bond_dimension is not None:
            options["matrix_product_state_max_bond_dimension"] = int(
                cfg.mps_max_bond_dimension
            )
        if cfg.mps_truncation_threshold is not None:
            options["matrix_product_state_truncation_threshold"] = float(
                cfg.mps_truncation_threshold
            )
        return AerSimulator(**options)

    @staticmethod
    def _counts_for_each_circuit(
        result: Any, n_circuits: int
    ) -> list[Mapping[str, int]]:
        if int(n_circuits) == 1:
            return [result.get_counts(0)]
        return [result.get_counts(index) for index in range(int(n_circuits))]

    def basis_circuit_count(self) -> int:
        """Return the number of randomized basis schedules executed per stream."""

        total_shots = int(self.shadow_config.shots)
        shots_per_basis = min(int(self.circuit_config.shots_per_basis), total_shots)
        return (total_shots + shots_per_basis - 1) // shots_per_basis

    def run_stream(
        self,
        inputs: Sequence[float] | np.ndarray,
        backend: Any | None = None,
        progress_callback: Callable[[int], None] | None = None,
    ) -> np.ndarray:
        """Execute sampled trajectories and reconstruct time-indexed shadow features."""

        values = np.asarray(inputs, dtype=float).reshape(-1)
        if values.size == 0:
            return np.empty((0, len(self.feature_names)), dtype=float)
        total_shots = int(self.shadow_config.shots)
        shots_per_basis = min(int(self.circuit_config.shots_per_basis), total_shots)
        n_schedules = self.basis_circuit_count()
        basis_schedules = sample_basis_schedules(
            n_schedules,
            values.size,
            self.n_readout,
            seed=self._rng_seed,
        )
        execution_backend = backend or self.backend
        owns_backend = execution_backend is None
        if execution_backend is None:
            execution_backend = self._aer_backend()
        if transpile is None:
            _require_qiskit()

        schedule_shots = np.full(n_schedules, shots_per_basis, dtype=int)
        schedule_shots[-1] = total_shots - shots_per_basis * (n_schedules - 1)
        expanded_schedules: list[np.ndarray] = []
        expanded_outcomes: list[np.ndarray] = []
        batch_size = int(self.circuit_config.circuit_batch_size)
        job_index = 0
        for shots in np.unique(schedule_shots):
            indices = np.flatnonzero(schedule_shots == shots)
            for batch_start in range(0, indices.size, batch_size):
                batch_indices = indices[batch_start : batch_start + batch_size]
                circuits = [
                    self.build_streaming_circuit(values, basis_schedules[index])
                    for index in batch_indices
                ]
                executable = transpile(
                    circuits,
                    backend=execution_backend,
                    optimization_level=int(
                        self.circuit_config.transpile_optimization_level
                    ),
                )
                run_options: dict[str, Any] = {"shots": int(shots)}
                if owns_backend:
                    run_options["seed_simulator"] = self._rng_seed + job_index
                result = execution_backend.run(executable, **run_options).result()
                for index, counts in zip(
                    batch_indices,
                    self._counts_for_each_circuit(result, batch_indices.size),
                ):
                    group_outcomes = counts_to_outcomes(
                        counts,
                        shots=int(shots),
                        n_steps=values.size,
                        n_readout=self.n_readout,
                    )
                    expanded_schedules.append(
                        np.repeat(
                            basis_schedules[int(index)][None, :, :], int(shots), axis=0
                        )
                    )
                    expanded_outcomes.append(group_outcomes)
                if progress_callback is not None:
                    progress_callback(int(batch_indices.size))
                job_index += 1

        schedules = np.concatenate(expanded_schedules, axis=0)
        outcomes = np.concatenate(expanded_outcomes, axis=0)

        features = estimate_shadow_features(
            schedules,
            outcomes,
            self.pauli_labels,
            include_bias=bool(self.shadow_config.include_bias),
        )
        if self.shadow_config.return_raw_shots:
            self.last_basis_schedules = schedules
            self.last_outcomes = outcomes
        else:
            self.last_basis_schedules = None
            self.last_outcomes = None
        return features

    def run(self, inputs: Sequence[float] | np.ndarray) -> np.ndarray:
        """Reservoir protocol alias."""

        return self.run_stream(inputs)

    def transform(self, inputs: Sequence[float] | np.ndarray) -> np.ndarray:
        """Scikit-style alias."""

        return self.run_stream(inputs)

    def get_feature_names(self) -> list[str]:
        """Return readout Pauli feature names."""

        return list(self.feature_names)

    @property
    def feature_labels(self) -> list[str]:
        """Compatibility alias used by pyqres examples."""

        return self.get_feature_names()


__all__ = ["QiskitGlobalFloquetPartialShadowReservoir"]
