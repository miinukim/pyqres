"""Dense exact quantum reservoir core.

This module is the numerical center of `pyqres.simulation`. It constructs dense
Hamiltonians/unitaries for small QRC systems, applies ancilla measurement
protocols exactly through Kraus operators, and exposes reduced memory channels
for both task-side reservoirs and PTM/dimension analysis.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Sequence

import numpy as np
from scipy.linalg import expm

from pyqres.core.control import (
    MeasurementControlConfig,
    embed_single_qubit_gate,
    projective_measurement_kraus,
    single_qubit_gate,
    weak_measurement_kraus,
)
from pyqres.core.reservoir_params import ReservoirParams


PAULI_1Q = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _kron_all(ops: Sequence[np.ndarray]) -> np.ndarray:
    """Kronecker product of a sequence of local operators."""

    out = np.array([[1.0 + 0.0j]])
    for op in ops:
        out = np.kron(out, op)
    return out


def _single_pauli_dense(n_qubits: int, target: int, pauli: str) -> np.ndarray:
    ops = [PAULI_1Q["I"]] * n_qubits
    ops[target] = PAULI_1Q[pauli]
    return _kron_all(ops)


def _two_pauli_dense(n_qubits: int, q1: int, p1: str, q2: int, p2: str) -> np.ndarray:
    ops = [PAULI_1Q["I"]] * n_qubits
    ops[q1] = PAULI_1Q[p1]
    ops[q2] = PAULI_1Q[p2]
    return _kron_all(ops)


def partial_trace_ancilla(op: np.ndarray, dim_system: int, dim_ancilla: int) -> np.ndarray:
    """Trace out the last subsystem, interpreted here as the ancilla register."""

    return np.trace(op.reshape(dim_system, dim_ancilla, dim_system, dim_ancilla), axis1=1, axis2=3)


def computational_zero_density(n_qubits: int) -> np.ndarray:
    ket = np.zeros((2**n_qubits, 1), dtype=complex)
    ket[0, 0] = 1.0
    return ket @ ket.conj().T


def _bits_to_int(bits: Sequence[int]) -> int:
    out = 0
    for bit in bits:
        out = (out << 1) | int(bit)
    return out


def _int_to_bits(value: int, n_qubits: int) -> list[int]:
    return [(value >> (n_qubits - 1 - idx)) & 1 for idx in range(n_qubits)]


def _embed_local_unitary(n_qubits: int, targets: Sequence[int], local_unitary: np.ndarray) -> np.ndarray:
    """Embed a k-qubit unitary acting on arbitrary target indices.

    The routine is intentionally explicit rather than clever: it loops through
    computational-basis columns, substitutes the target bits, and writes the
    corresponding amplitudes. This keeps target-index semantics auditable for
    small dense systems.
    """

    targets = tuple(int(t) for t in targets)
    if len(targets) == 0:
        return np.eye(2**n_qubits, dtype=complex)
    if len(set(targets)) != len(targets):
        raise ValueError("targets must be unique")
    if any(t < 0 or t >= n_qubits for t in targets):
        raise ValueError(f"targets {targets} are out of range for n_qubits={n_qubits}")

    local_unitary = np.asarray(local_unitary, dtype=complex)
    local_dim = 2 ** len(targets)
    if local_unitary.shape != (local_dim, local_dim):
        raise ValueError(
            f"Local unitary has shape {local_unitary.shape}, expected {(local_dim, local_dim)} "
            f"for {len(targets)} target qubits."
        )

    dim = 2**n_qubits
    out = np.zeros((dim, dim), dtype=complex)
    for col in range(dim):
        in_bits = _int_to_bits(col, n_qubits)
        local_col = _bits_to_int([in_bits[t] for t in targets])
        for local_row in range(local_dim):
            amp = local_unitary[local_row, local_col]
            if amp == 0.0:
                continue
            out_bits = list(in_bits)
            replacement = _int_to_bits(local_row, len(targets))
            for idx, target in enumerate(targets):
                out_bits[target] = replacement[idx]
            row = _bits_to_int(out_bits)
            out[row, col] = amp
    return out


def _statevector_to_preparation_unitary(statevector: np.ndarray) -> np.ndarray:
    """Build a unitary whose first column prepares `statevector` from |0...0>."""

    vec = np.asarray(statevector, dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-15:
        raise ValueError("State-preparation vector must have non-zero norm.")
    vec = vec / norm
    dim = vec.shape[0]
    basis = np.eye(dim, dtype=complex)
    basis[:, 0] = vec
    q, _ = np.linalg.qr(basis)
    phase = np.vdot(vec, q[:, 0])
    if abs(phase) > 1e-15:
        q[:, 0] *= phase.conjugate() / abs(phase)
    return q


def _coerce_unitary_matrix(unitary_like: Any) -> np.ndarray:
    """Accept NumPy/Qiskit/unitary-like objects and return a validated matrix."""

    if isinstance(unitary_like, np.ndarray):
        matrix = np.asarray(unitary_like, dtype=complex)
    else:
        try:
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Operator
        except Exception:
            QuantumCircuit = None  # type: ignore
            Operator = None  # type: ignore

        if Operator is not None and isinstance(unitary_like, Operator):
            matrix = np.asarray(unitary_like.data, dtype=complex)
        elif QuantumCircuit is not None and isinstance(unitary_like, QuantumCircuit):
            matrix = np.asarray(Operator(unitary_like).data, dtype=complex)
        elif hasattr(unitary_like, "to_matrix"):
            matrix = np.asarray(unitary_like.to_matrix(), dtype=complex)
        else:
            raise TypeError(
                "Unsupported input unitary. Provide a NumPy array, a qiskit QuantumCircuit/Operator, "
                "or an object exposing to_matrix()."
            )

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Input unitary must be square, got shape {matrix.shape}.")
    eye = np.eye(matrix.shape[0], dtype=complex)
    if not np.allclose(matrix.conj().T @ matrix, eye, atol=1e-10):
        raise ValueError("Input unitary must be unitary.")
    return matrix


@dataclass
class ExactQRCModelConfig:
    """Configuration for dense exact QRC evolution.

    The scalar input can enter as Hamiltonian modulation, amplitude preparation,
    or a user-provided unitary. Measurement behavior is delegated to
    `MeasurementControlConfig` so the same control semantics can be shared with
    frontends and analysis wrappers.
    """

    n_system: int = 4
    n_ancilla: int = 2
    tau: float = 1.0
    input_encoding: Literal["hamiltonian", "amplitude", "unitary"] = "hamiltonian"
    input_scale: float = 1.0
    input_bias: float = 0.0
    encoding_register: Literal["ancilla", "system", "full"] = "ancilla"
    encoding_targets: tuple[int, ...] = (0,)
    amplitude_encoding_style: Literal["u_sqrt_1_minus_u", "sqrt_u_sqrt_1_minus_u"] = "u_sqrt_1_minus_u"
    amplitude_state_factory: Optional[Callable[[float], Sequence[complex] | np.ndarray]] = None
    input_unitary_factory: Optional[Callable[[float], Any]] = None
    input_unitary_order: Literal["before", "after", "replace"] = "before"
    unitary_cache_size: Optional[int] = 128
    hx0_vec: Optional[np.ndarray] = None
    hz1_vec: Optional[np.ndarray] = None
    J_mat: Optional[np.ndarray] = None
    seed: int = 17462
    connectivity_kind: str = "full"
    control: MeasurementControlConfig = field(default_factory=MeasurementControlConfig)


class ExactQRCModel:
    """Dense exact QRC core shared by pyqres execution and dimension analysis.

    The shared semantics are:
    - input encoding through one of:
      - Hamiltonian modulation: H(u) = H0 + input_scale * u * H1
      - amplitude preparation on selected qubits through an input-dependent unitary
      - a user-supplied input-dependent unitary (NumPy or qiskit)
    - exact dense evolution under a joint unitary U(u)
    - ancilla measurement using projective or weak Kraus operators
    - optional output-conditioned gates after measurement
    - post-measurement ancilla reset or keep
    """

    def __init__(self, cfg: ExactQRCModelConfig):
        self.cfg = cfg
        self.nS = int(cfg.n_system)
        self.nA = int(cfg.n_ancilla)
        self.n = self.nS + self.nA
        if self.nA <= 0:
            raise ValueError("n_ancilla must be >= 1.")
        self.dim_system = 2**self.nS
        self.dim_ancilla = 2**self.nA
        self.dim_total = 2**self.n
        self.control = cfg.control.validated(self.nS, self.nA)

        if cfg.hx0_vec is None or cfg.hz1_vec is None or cfg.J_mat is None:
            # If explicit Hamiltonian parameters are absent, derive a
            # reproducible random reservoir from the compact ReservoirParams
            # generator. Explicit arrays always take precedence.
            generated = ReservoirParams(
                n_system=self.nS,
                n_ancilla=self.nA,
                tau=cfg.tau,
                seed=cfg.seed,
                graph_kind=cfg.connectivity_kind,
            ).generate()
            self.hx0 = np.asarray(generated["hx0_vec"], dtype=float)
            self.hz1 = np.asarray(generated["hz1_vec"], dtype=float)
            self.J = np.asarray(generated["J_mat"], dtype=float)
            self.tau = float(generated["tau"])
        else:
            self.hx0 = np.asarray(cfg.hx0_vec, dtype=float)
            self.hz1 = np.asarray(cfg.hz1_vec, dtype=float)
            self.J = np.asarray(cfg.J_mat, dtype=float)
            self.tau = float(cfg.tau)

        self.ancilla_reset_density = computational_zero_density(self.nA)
        self._unitary_cache: OrderedDict[float, np.ndarray] = OrderedDict()
        self._measurement_kraus = self._build_measurement_kraus()
        self._identity_system = np.eye(self.dim_system, dtype=complex)
        self.H0, self.H1 = self._build_h0_h1()
        self._fixed_dynamics_unitary = expm(-1j * self.tau * self.H0)

    def clear_caches(self) -> None:
        """Release cached dense unitary arrays held by this model."""

        self._unitary_cache.clear()

    def _cache_unitary(self, key: float, value: np.ndarray) -> None:
        """Store a dense unitary in an LRU-style cache keyed by scalar input."""

        self._unitary_cache[key] = value
        self._unitary_cache.move_to_end(key)
        max_entries = self.cfg.unitary_cache_size
        if max_entries is not None:
            max_entries = int(max_entries)
            if max_entries <= 0:
                self._unitary_cache.clear()
                return
            while len(self._unitary_cache) > max_entries:
                self._unitary_cache.popitem(last=False)

    def _build_h0_h1(self) -> tuple[np.ndarray, np.ndarray]:
        """Build fixed and input-modulated Hamiltonian components.

        `H0` contains transverse fields plus ZZ couplings. `H1` contains the
        longitudinal Z field that is scaled by the scalar input under
        Hamiltonian encoding.
        """

        h0 = np.zeros((self.dim_total, self.dim_total), dtype=complex)
        h1 = np.zeros_like(h0)
        for idx in range(self.n):
            h0 += float(self.hx0[idx]) * _single_pauli_dense(self.n, idx, "X")
            h1 += float(self.hz1[idx]) * _single_pauli_dense(self.n, idx, "Z")
        for i in range(self.n):
            for j in range(i + 1, self.n):
                jij = float(self.J[i, j])
                if jij != 0.0:
                    h0 += jij * _two_pauli_dense(self.n, i, "Z", j, "Z")
        return h0, h1

    def _build_measurement_kraus(self) -> list[np.ndarray]:
        """Create ancilla-only Kraus operators from the configured measurement."""

        if self.control.measurement_mode == "projective":
            return projective_measurement_kraus(self.nA)
        return weak_measurement_kraus(self.nA, float(self.control.measurement_strength))

    def _encoding_targets_global(self) -> tuple[int, ...]:
        """Map local system/ancilla target indices into joint-register indices."""

        targets = tuple(int(t) for t in self.cfg.encoding_targets)
        register = self.cfg.encoding_register
        if register == "full":
            global_targets = targets
        elif register == "system":
            if any(t < 0 or t >= self.nS for t in targets):
                raise ValueError(f"encoding_targets {targets} must lie in the system register [0, {self.nS}).")
            global_targets = targets
        elif register == "ancilla":
            if any(t < 0 or t >= self.nA for t in targets):
                raise ValueError(f"encoding_targets {targets} must lie in the ancilla register [0, {self.nA}).")
            global_targets = tuple(self.nS + t for t in targets)
        else:
            raise ValueError(f"Unsupported encoding_register '{register}'.")

        if any(t < 0 or t >= self.n for t in global_targets):
            raise ValueError(f"encoding_targets {targets} are out of range for total qubits={self.n}.")
        return global_targets

    def _scaled_input(self, u: float) -> float:
        """Apply affine input scaling used by non-Hamiltonian encodings."""

        return float(self.cfg.input_bias + self.cfg.input_scale * float(u))

    def _amplitude_statevector(self, u: float, n_targets: int) -> np.ndarray:
        """Create the target-register state used for amplitude encoding."""

        mapped_u = float(np.clip(self._scaled_input(u), 0.0, 1.0))
        if self.cfg.amplitude_state_factory is not None:
            state = np.asarray(self.cfg.amplitude_state_factory(mapped_u), dtype=complex).reshape(-1)
        else:
            if self.cfg.amplitude_encoding_style == "u_sqrt_1_minus_u":
                local_state = np.array([mapped_u, np.sqrt(max(0.0, 1.0 - mapped_u))], dtype=complex)
            elif self.cfg.amplitude_encoding_style == "sqrt_u_sqrt_1_minus_u":
                local_state = np.array([np.sqrt(mapped_u), np.sqrt(max(0.0, 1.0 - mapped_u))], dtype=complex)
            else:
                raise ValueError(f"Unsupported amplitude_encoding_style '{self.cfg.amplitude_encoding_style}'.")
            state = local_state
            for _ in range(1, n_targets):
                state = np.kron(state, local_state)

        expected_dim = 2**n_targets
        if state.shape != (expected_dim,):
            raise ValueError(
                f"Amplitude state has length {state.shape[0]}, expected {expected_dim} "
                f"for {n_targets} target qubits."
            )
        return state / max(np.linalg.norm(state), 1e-15)

    def _input_unitary_local(self, u: float, n_targets: int) -> np.ndarray:
        """Return the local input unitary before embedding into the full system."""

        if self.cfg.input_encoding == "amplitude":
            state = self._amplitude_statevector(u, n_targets)
            return _statevector_to_preparation_unitary(state)
        if self.cfg.input_encoding == "unitary":
            if self.cfg.input_unitary_factory is None:
                raise ValueError("input_unitary_factory must be provided when input_encoding='unitary'.")
            unitary_like = self.cfg.input_unitary_factory(float(self._scaled_input(u)))
            matrix = _coerce_unitary_matrix(unitary_like)
            expected_dim = 2**n_targets
            if matrix.shape != (expected_dim, expected_dim):
                raise ValueError(
                    f"Input unitary has shape {matrix.shape}, expected {(expected_dim, expected_dim)} "
                    f"for targets {self.cfg.encoding_targets}."
                )
            return matrix
        raise ValueError(f"_input_unitary_local does not support input_encoding='{self.cfg.input_encoding}'.")

    def encoding_unitary(self, u: float) -> np.ndarray:
        """Construct the full-register input-encoding unitary."""

        if self.cfg.input_encoding == "hamiltonian":
            return np.eye(self.dim_total, dtype=complex)

        targets = self._encoding_targets_global()
        local = self._input_unitary_local(u, len(targets))
        if self.cfg.encoding_register == "full":
            if targets != tuple(range(self.n)):
                raise ValueError(
                    "encoding_register='full' requires encoding_targets to cover the full register in order, "
                    f"got {self.cfg.encoding_targets}."
                )
            return local
        return _embed_local_unitary(self.n, targets, local)

    def unitary(self, u: float) -> np.ndarray:
        """Return the joint evolution unitary for scalar input `u`."""

        key = float(u)
        cached = self._unitary_cache.get(key)
        if cached is not None:
            self._unitary_cache.move_to_end(key)
            return cached
        if self.cfg.input_encoding == "hamiltonian":
            # Hermitize defensively before exponentiation; generated H0/H1 are
            # Hermitian, but this guards against small user-array asymmetries.
            h = self.H0 + (self.cfg.input_scale * key) * self.H1
            h = 0.5 * (h + h.conj().T)
            out = expm(-1j * self.tau * h)
        else:
            encoding = self.encoding_unitary(key)
            if self.cfg.input_unitary_order == "before":
                out = self._fixed_dynamics_unitary @ encoding
            elif self.cfg.input_unitary_order == "after":
                out = encoding @ self._fixed_dynamics_unitary
            elif self.cfg.input_unitary_order == "replace":
                out = encoding
            else:
                raise ValueError(f"Unsupported input_unitary_order '{self.cfg.input_unitary_order}'.")
        self._cache_unitary(key, out)
        return out

    def zero_system_state(self) -> np.ndarray:
        return computational_zero_density(self.nS)

    def maximally_mixed_system_state(self) -> np.ndarray:
        return np.eye(self.dim_system, dtype=complex) / float(self.dim_system)

    def initial_system_density(self, init_state: str = "maximally_mixed") -> np.ndarray:
        if init_state == "zero":
            return self.zero_system_state()
        return self.maximally_mixed_system_state()

    def initial_joint_density(self, init_state: str = "maximally_mixed") -> np.ndarray:
        return np.kron(self.initial_system_density(init_state), self.ancilla_reset_density)

    def evolve_joint(self, rho_joint: np.ndarray, u: float) -> np.ndarray:
        """Apply the joint unitary to a system+ancilla operator or state."""

        uop = self.unitary(float(u))
        return uop @ rho_joint @ uop.conj().T

    def _condition_matches(self, outcome: int) -> bool:
        """Evaluate the classical feedback condition for one measurement outcome."""

        if self.control.conditioned_gate_condition == "all_one":
            return outcome == (2**self.nA - 1)
        return outcome != 0

    def conditioned_gate(self, outcome: int) -> np.ndarray:
        """Return the full-register feedback gate for a measurement outcome."""

        gate_name = self.control.conditioned_gate
        if gate_name == "none" or not self._condition_matches(outcome):
            return np.eye(self.dim_total, dtype=complex)
        angle = float(self.control.conditioned_gate_angle)
        target = int(self.control.conditioned_gate_target)
        kind = gate_name.split("_", 1)[1]
        local_gate = single_qubit_gate(kind, angle)
        if gate_name.startswith("system_"):
            return embed_single_qubit_gate(self.n, target, local_gate)
        if gate_name.startswith("ancilla_"):
            return embed_single_qubit_gate(self.n, self.nS + target, local_gate)
        raise ValueError(f"Unsupported conditioned_gate '{gate_name}'.")

    def measurement_branches(self, rho_joint: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Return unnormalized post-measurement branches and their probabilities."""

        probs = np.zeros(self.dim_ancilla, dtype=float)
        branches: list[np.ndarray] = []
        for outcome, kraus_ancilla in enumerate(self._measurement_kraus):
            full_kraus = np.kron(self._identity_system, kraus_ancilla)
            branch = full_kraus @ rho_joint @ full_kraus.conj().T
            prob = max(float(np.real_if_close(np.trace(branch))), 0.0)
            probs[outcome] = prob
            branches.append(branch)
        total = float(probs.sum())
        if total <= 1e-18:
            probs[:] = 1.0 / float(self.dim_ancilla)
            branches = [rho_joint.copy() / float(self.dim_ancilla) for _ in range(self.dim_ancilla)]
        return probs, branches

    def apply_measurement_protocol_exact(self, rho_joint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply measurement, optional feedback, normalization, and reset exactly."""

        probs = np.zeros(self.dim_ancilla, dtype=float)
        post = np.zeros_like(rho_joint, dtype=complex)
        for outcome, kraus_ancilla in enumerate(self._measurement_kraus):
            full_kraus = np.kron(self._identity_system, kraus_ancilla)
            branch = full_kraus @ rho_joint @ full_kraus.conj().T
            probs[outcome] = max(float(np.real_if_close(np.trace(branch))), 0.0)
            if self.control.conditioned_gate == "none" or not self._condition_matches(outcome):
                post += branch
            else:
                gate = self.conditioned_gate(outcome)
                post += gate @ branch @ gate.conj().T

        total = float(probs.sum())
        if total <= 1e-18:
            probs[:] = 1.0 / float(self.dim_ancilla)
            post = rho_joint.copy()
        else:
            probs /= total
            post /= total

        if self.control.post_measurement_mode == "reset":
            # Resetting is implemented by tracing out the measured ancilla and
            # tensoring the reduced memory state with |0...0><0...0|.
            rho_system = partial_trace_ancilla(post, self.dim_system, self.dim_ancilla)
            next_joint = np.kron(rho_system, self.ancilla_reset_density)
        else:
            next_joint = post
        return probs, next_joint

    def sample_measurement_protocol(self, rho_joint: np.ndarray, rng: np.random.Generator) -> tuple[int, np.ndarray]:
        """Sample one measurement branch and return the normalized next state."""

        probs = np.zeros(self.dim_ancilla, dtype=float)
        for outcome, kraus_ancilla in enumerate(self._measurement_kraus):
            full_kraus = np.kron(self._identity_system, kraus_ancilla)
            branch = full_kraus @ rho_joint @ full_kraus.conj().T
            probs[outcome] = max(float(np.real_if_close(np.trace(branch))), 0.0)

        total = float(probs.sum())
        if total <= 1e-18:
            probs[:] = 1.0 / float(self.dim_ancilla)
        else:
            probs /= total
        outcome = int(rng.choice(np.arange(self.dim_ancilla), p=probs))
        full_kraus = np.kron(self._identity_system, self._measurement_kraus[outcome])
        branch = full_kraus @ rho_joint @ full_kraus.conj().T
        norm = max(float(np.real_if_close(np.trace(branch))), 1e-18)
        post = branch / norm
        gate = self.conditioned_gate(outcome)
        post = gate @ post @ gate.conj().T
        if self.control.post_measurement_mode == "reset":
            rho_system = partial_trace_ancilla(post, self.dim_system, self.dim_ancilla)
            next_joint = np.kron(rho_system, self.ancilla_reset_density)
        else:
            next_joint = post
        return outcome, next_joint

    def system_channel(self, u: float, op_system: np.ndarray) -> np.ndarray:
        """Apply the induced memory channel to an arbitrary system operator."""

        if self.control.post_measurement_mode != "reset":
            raise NotImplementedError("system_channel requires post_measurement_mode='reset'.")
        joint = np.kron(np.asarray(op_system, dtype=complex), self.ancilla_reset_density)
        evolved = self.evolve_joint(joint, float(u))
        _, next_joint = self.apply_measurement_protocol_exact(evolved)
        return partial_trace_ancilla(next_joint, self.dim_system, self.dim_ancilla)

    def exact_step_from_system(self, rho_system: np.ndarray, u: float) -> tuple[np.ndarray, np.ndarray]:
        """Advance a reduced memory state and return readout probabilities."""

        if self.control.post_measurement_mode != "reset":
            raise NotImplementedError("exact_step_from_system requires post_measurement_mode='reset'.")
        joint = np.kron(np.asarray(rho_system, dtype=complex), self.ancilla_reset_density)
        evolved = self.evolve_joint(joint, float(u))
        probs, next_joint = self.apply_measurement_protocol_exact(evolved)
        rho_next = partial_trace_ancilla(next_joint, self.dim_system, self.dim_ancilla)
        return probs, rho_next
