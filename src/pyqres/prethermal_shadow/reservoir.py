from __future__ import annotations

"""Global-Floquet reservoir with partial classical-shadow readout."""

from collections.abc import Sequence
from itertools import combinations, product

import numpy as np
import scipy.linalg as la

from .config import GlobalFloquetConfig, InputEncodingConfig, PartialShadowReadoutConfig, ReadoutResetConfig
from .dynamics import (
    build_drive_hamiltonian,
    build_global_h0,
    build_step_unitary,
    density_plus,
    density_zero,
    generate_hamiltonian_parameters,
    kron_all,
    partial_trace_memory_first,
    pauli_string,
    parse_pauli_operator,
    project_density,
    step_duration,
    validate_floquet_config,
)
from .shadows import (
    exact_readout_expectations,
    generate_pauli_labels,
    partial_shadow_feature_names,
    sample_partial_shadow_features,
)


def _validate_axis(axis: str, name: str) -> str:
    out = str(axis).upper()
    if out not in {"X", "Y", "Z"}:
        raise ValueError(f"{name} must be x, y, or z.")
    return out


def _resolve_input_qubits(cfg: InputEncodingConfig, floquet: GlobalFloquetConfig) -> tuple[int, ...]:
    spec = cfg.input_qubits
    if isinstance(spec, str):
        key = spec.lower()
        if key == "memory":
            return tuple(range(int(floquet.n_memory)))
        if key == "all":
            return tuple(range(int(floquet.n_qubits)))
        raise ValueError("input_qubits must be 'memory', 'all', or a sequence of indices.")
    qubits = tuple(int(q) for q in spec)
    if not qubits:
        raise ValueError("input_qubits sequence must be non-empty.")
    if any(q < 0 or q >= int(floquet.n_qubits) for q in qubits):
        raise ValueError("input_qubits contains an out-of-range qubit.")
    return qubits


def _beta_array(cfg: InputEncodingConfig, n_targets: int) -> np.ndarray:
    if isinstance(cfg.beta, (int, float, np.floating)):
        base = np.full(int(n_targets), float(cfg.beta), dtype=float)
    else:
        base = np.asarray(cfg.beta, dtype=float).reshape(-1)
        if base.shape != (int(n_targets),):
            raise ValueError(f"beta must be scalar or length {n_targets}.")
    if not cfg.random_beta:
        return base
    rng = np.random.default_rng(int(cfg.seed))
    return base * rng.uniform(0.5, 1.5, size=int(n_targets))


def _low_weight_labels(n_qubits: int, pauli_k: int) -> list[tuple[tuple[int, str], ...]]:
    labels = []
    for weight in range(1, int(pauli_k) + 1):
        for sites in combinations(range(int(n_qubits)), weight):
            for paulis in product(("X", "Y", "Z"), repeat=weight):
                labels.append(tuple((int(site), str(pauli)) for site, pauli in zip(sites, paulis)))
    return labels


class GlobalFloquetPartialShadowReservoir:
    """Dense global-Floquet QRC with partial local Pauli-shadow readout."""

    def __init__(
        self,
        floquet_config: GlobalFloquetConfig,
        input_config: InputEncodingConfig | None = None,
        shadow_config: PartialShadowReadoutConfig | None = None,
        reset_config: ReadoutResetConfig | None = None,
        simulator_method: str = "density_matrix",
        seed_simulator: int | None = None,
    ):
        validate_floquet_config(floquet_config)
        if simulator_method != "density_matrix":
            raise ValueError("GlobalFloquetPartialShadowReservoir currently requires simulator_method='density_matrix'.")
        self.floquet_config = floquet_config
        self.input_config = input_config or InputEncodingConfig()
        self.shadow_config = shadow_config or PartialShadowReadoutConfig()
        self.reset_config = reset_config or ReadoutResetConfig()
        self.simulator_method = simulator_method
        self.seed_simulator = seed_simulator

        self.n_qubits = int(floquet_config.n_qubits)
        self.n_memory = int(floquet_config.n_memory)
        self.n_readout = int(floquet_config.n_readout)
        self.dim_memory = 2**self.n_memory
        self.dim_readout = 2**self.n_readout
        self.dim_total = 2**self.n_qubits
        self.delta_t = step_duration(floquet_config)

        if int(self.shadow_config.pauli_k) < 1 or int(self.shadow_config.pauli_k) > self.n_readout:
            raise ValueError("shadow pauli_k must lie in [1, n_readout].")
        if int(self.shadow_config.shots) <= 0:
            raise ValueError("shadow shots must be positive.")
        if str(self.shadow_config.measurement_type).lower() not in {"projective", "weak"}:
            raise ValueError("measurement_type must be projective or weak.")
        if not (0.0 < float(self.shadow_config.weak_strength) <= 1.0):
            raise ValueError("weak_strength must lie in (0, 1].")
        if self.shadow_config.basis_randomization != "local_pauli":
            raise ValueError("only basis_randomization='local_pauli' is supported.")
        if not self.reset_config.reset_after_measurement or not self.reset_config.trace_readout_after_step:
            raise NotImplementedError("only trace/reset readout updates are implemented.")
        if self.reset_config.reset_state not in {"zero", "plus"}:
            raise ValueError("reset_state must be 'zero' or 'plus'.")

        params = generate_hamiltonian_parameters(floquet_config)
        self.edges = params["edges"]
        self.h = np.asarray(params["h"], dtype=float)
        self.jz = np.asarray(params["jz"], dtype=float)
        self.jxy = np.asarray(params["jxy"], dtype=float)
        self.drive_coeffs = np.asarray(params["drive_coeffs"], dtype=float)
        self.break_coeffs = np.asarray(params["break_coeffs"], dtype=float)
        self.h0 = build_global_h0(
            floquet_config,
            h=self.h,
            jz=self.jz,
            jxy=self.jxy,
            break_coeffs=self.break_coeffs,
            edges=self.edges,
        )
        self.drive_hamiltonian = build_drive_hamiltonian(floquet_config, self.drive_coeffs)
        self.u_floquet_step = build_step_unitary(floquet_config, self.h0, self.drive_hamiltonian)

        self.input_operator: np.ndarray | None = None
        if self.input_config.operator is None:
            self.input_qubits = _resolve_input_qubits(self.input_config, floquet_config)
            self.input_axis = _validate_axis(self.input_config.axis, "input axis")
            self.beta = _beta_array(self.input_config, len(self.input_qubits))
        else:
            if not isinstance(self.input_config.beta, (int, float, np.floating)):
                raise ValueError("operator input encoding requires scalar beta.")
            self.input_qubits = tuple()
            self.input_axis = _validate_axis(self.input_config.axis, "input axis")
            self.beta = np.asarray([float(self.input_config.beta)], dtype=float)
            self.input_operator = parse_pauli_operator(
                self.n_qubits,
                str(self.input_config.operator),
                normalize=bool(self.input_config.normalize_operator),
            )
        self.pauli_labels = generate_pauli_labels(self.n_readout, int(self.shadow_config.pauli_k))
        self.feature_names = partial_shadow_feature_names(
            self.n_readout,
            int(self.shadow_config.pauli_k),
            include_bias=bool(self.shadow_config.include_bias),
        )
        self._rng_seed = int(self.shadow_config.seed if seed_simulator is None else seed_simulator)
        self.last_raw_shots: dict[str, np.ndarray] | None = None
        self.reset_state()

    def reset_state(self) -> None:
        """Reset persistent memory state and the shadow RNG."""

        self.rho_memory = density_zero(self.n_memory)
        self._shadow_rng = np.random.default_rng(self._rng_seed)
        self.last_raw_shots = None

    def _readout_reset_state(self) -> np.ndarray:
        if self.reset_config.reset_state == "zero":
            return density_zero(self.n_readout)
        return density_plus(self.n_readout)

    def _input_unitary(self, u: float) -> np.ndarray:
        scale = float(u) + float(self.input_config.bias)
        if self.input_operator is not None:
            return la.expm(-1j * float(self.beta[0]) * scale * self.input_operator)
        out = np.eye(self.dim_total, dtype=complex)
        for q, beta in zip(self.input_qubits, self.beta):
            generator = pauli_string(self.n_qubits, [(int(q), self.input_axis)])
            out = la.expm(-1j * float(beta) * scale * generator) @ out
        return out

    def _pre_measurement_state(self, rho_memory: np.ndarray, u: float) -> np.ndarray:
        rho0 = np.kron(project_density(rho_memory), self._readout_reset_state())
        u_step = self.u_floquet_step @ self._input_unitary(float(u))
        rho_pre = u_step @ rho0 @ u_step.conj().T
        return project_density(rho_pre)

    def exact_features_from_state(self, rho_pre: np.ndarray) -> np.ndarray:
        """Compute exact readout Pauli expectations from a pre-reset full state."""

        rho_r = partial_trace_memory_first(rho_pre, self.n_memory, self.n_readout, keep="readout")
        return exact_readout_expectations(
            project_density(rho_r),
            self.pauli_labels,
            include_bias=bool(self.shadow_config.include_bias),
        )

    def shadow_features_from_state(self, rho_pre: np.ndarray, shots: int) -> np.ndarray:
        """Simulate partial local Pauli shadow feature estimates on readout."""

        rho_r = partial_trace_memory_first(rho_pre, self.n_memory, self.n_readout, keep="readout")
        result = sample_partial_shadow_features(
            project_density(rho_r),
            self.pauli_labels,
            shots=int(shots),
            measurement_type=str(self.shadow_config.measurement_type),
            weak_strength=float(self.shadow_config.weak_strength),
            include_bias=bool(self.shadow_config.include_bias),
            rng=self._shadow_rng,
            return_raw=bool(self.shadow_config.return_raw_shots),
        )
        if isinstance(result, tuple):
            features, raw = result
            self.last_raw_shots = raw
            return np.asarray(features, dtype=float)
        self.last_raw_shots = None
        return np.asarray(result, dtype=float)

    def step(self, u: float) -> np.ndarray:
        """Apply one reservoir step and return one feature vector."""

        rho_pre = self._pre_measurement_state(self.rho_memory, float(u))
        if self.shadow_config.exact_expectations or not self.shadow_config.return_shadow_estimates:
            features = self.exact_features_from_state(rho_pre)
        else:
            features = self.shadow_features_from_state(rho_pre, int(self.shadow_config.shots))
        self.rho_memory = project_density(partial_trace_memory_first(rho_pre, self.n_memory, self.n_readout, keep="memory"))
        return np.asarray(features, dtype=float)

    def run(self, inputs: Sequence[float] | np.ndarray) -> np.ndarray:
        """Run over a scalar input sequence and return a feature matrix."""

        values = np.asarray(inputs, dtype=float).reshape(-1)
        self.reset_state()
        if values.size == 0:
            return np.empty((0, len(self.feature_names)), dtype=float)
        return np.vstack([self.step(float(value)) for value in values])

    def run_stream(self, inputs: Sequence[float] | np.ndarray) -> np.ndarray:
        """pyqres reservoir protocol alias."""

        return self.run(inputs)

    def transform(self, inputs: Sequence[float] | np.ndarray) -> np.ndarray:
        """scikit-style alias."""

        return self.run(inputs)

    def get_feature_names(self) -> list[str]:
        """Return readout Pauli feature names."""

        return list(self.feature_names)

    @property
    def feature_labels(self) -> list[str]:
        """Compatibility alias used by pyqres examples."""

        return self.get_feature_names()

    def _memory_channel(self, op_memory: np.ndarray, u_bar: float = 0.0) -> np.ndarray:
        rho0 = np.kron(np.asarray(op_memory, dtype=complex), self._readout_reset_state())
        u_step = self.u_floquet_step @ self._input_unitary(float(u_bar))
        out = u_step @ rho0 @ u_step.conj().T
        return partial_trace_memory_first(out, self.n_memory, self.n_readout, keep="memory")

    def build_projected_memory_channel(self, pauli_k: int = 2) -> np.ndarray:
        """Build projected induced memory channel on low-weight memory Paulis."""

        labels = _low_weight_labels(self.n_memory, int(pauli_k))
        ops = [pauli_string(self.n_memory, label) for label in labels]
        dim = self.dim_memory
        mat = np.zeros((len(ops), len(ops)), dtype=complex)
        for nu, op_in in enumerate(ops):
            evolved = self._memory_channel(op_in)
            for mu, op_out in enumerate(ops):
                mat[mu, nu] = np.trace(op_out @ evolved) / dim
        return mat

    def memory_channel_spectrum(self, pauli_k: int = 2) -> np.ndarray:
        """Return eigenvalues of the projected induced memory channel."""

        return np.linalg.eigvals(self.build_projected_memory_channel(pauli_k=pauli_k))

    def _scope_expectations(self, rho_pre: np.ndarray, scope: str, pauli_k: int) -> np.ndarray:
        scope_key = str(scope).lower()
        if scope_key == "full":
            rho = rho_pre
            n = self.n_qubits
        elif scope_key == "memory":
            rho = partial_trace_memory_first(rho_pre, self.n_memory, self.n_readout, keep="memory")
            n = self.n_memory
        elif scope_key == "readout":
            rho = partial_trace_memory_first(rho_pre, self.n_memory, self.n_readout, keep="readout")
            n = self.n_readout
        else:
            raise ValueError("observable_scope must be full, memory, or readout.")
        return np.asarray(
            [float(np.real_if_close(np.trace(pauli_string(n, label) @ rho))) for label in _low_weight_labels(n, int(pauli_k))],
            dtype=float,
        )

    def branch_sensitivity(
        self,
        u0: float,
        delta: float,
        horizon: int,
        observable_scope: str = "readout",
        pauli_k: int = 2,
    ) -> np.ndarray:
        """Distinguish two branches initialized by u0 +/- delta, then zero input."""

        if int(horizon) <= 0:
            return np.empty((0,), dtype=float)
        rho_plus = density_zero(self.n_memory)
        rho_minus = density_zero(self.n_memory)
        out = []
        for t in range(int(horizon)):
            u_plus = float(u0) + float(delta) if t == 0 else 0.0
            u_minus = float(u0) - float(delta) if t == 0 else 0.0
            pre_plus = self._pre_measurement_state(rho_plus, u_plus)
            pre_minus = self._pre_measurement_state(rho_minus, u_minus)
            diff = self._scope_expectations(pre_plus, observable_scope, pauli_k) - self._scope_expectations(pre_minus, observable_scope, pauli_k)
            out.append(float(np.linalg.norm(diff)))
            rho_plus = project_density(partial_trace_memory_first(pre_plus, self.n_memory, self.n_readout, keep="memory"))
            rho_minus = project_density(partial_trace_memory_first(pre_minus, self.n_memory, self.n_readout, keep="memory"))
        return np.asarray(out, dtype=float)


__all__ = [
    "GlobalFloquetPartialShadowReservoir",
]
