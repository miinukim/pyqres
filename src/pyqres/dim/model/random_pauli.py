"""Random Pauli-circuit dimension model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..linalg_utils import ensure_finite
from ..pauli import computational_zero_state
from .base import ReservoirBase


@dataclass
class RandomPauliReservoirParameters:
    n_memory: int = 5
    n_readout: int = 1
    depth: int = 3
    seed: int = 1234
    input_qubit: int = 0
    encoding_qubits: int = 1
    ancilla_state: str = "zero"
    input_bias: float = 0.5
    input_scale: float = 0.5


class RandomPauliReservoirModel(ReservoirBase):
    """Finite-depth random Pauli-basis circuit reservoir with GHZ-like input encoding.

    The circuit is fixed once at initialization. Each layer applies independent
    random single-qubit SU(2) rotations, expanded in the Pauli basis, followed
    by a brickwork pattern of nearest-neighbor CNOT gates across the full
    memory+readout register.

    The scalar input u is not injected through the Hamiltonian. Instead, it
    is first mapped to a valid qubit-population parameter

        p(u) = clip(input_bias + input_scale * u, 0, 1),

    then prepares a contiguous block of encoding_qubits readout qubits in the
    normalized GHZ-like state

        sqrt(p(u)) |0...0> + sqrt(1-p(u)) |1...1>,

    starting at input_qubit. Any remaining readout qubits are initialized in
    the chosen ancilla state (currently only |0> is supported). After the
    fixed random circuit is applied, the readout subsystem is discarded exactly
    as in the other reservoir models. The affine map is convenient because the
    Volterra analysis in this repository expands the channel locally around
    u = 0.
    """

    def __init__(self, params: RandomPauliReservoirParameters):
        if params.n_readout < 1:
            raise ValueError("RandomPauliReservoirModel requires n_readout >= 1")
        if not (0 <= params.input_qubit < params.n_readout):
            raise ValueError(
                f"input_qubit={params.input_qubit} must lie in [0, {params.n_readout - 1}]"
            )
        if params.encoding_qubits < 1:
            raise ValueError("encoding_qubits must be at least 1")
        if params.input_qubit + params.encoding_qubits > params.n_readout:
            raise ValueError(
                "The GHZ-like encoding block must fit inside the readout register: "
                f"input_qubit={params.input_qubit}, encoding_qubits={params.encoding_qubits}, "
                f"n_readout={params.n_readout}"
            )
        if params.depth < 1:
            raise ValueError("depth must be at least 1")
        if params.ancilla_state.lower() != "zero":
            raise ValueError("Only ancilla_state='zero' is currently supported")

        self.params = params
        self._initialize_common(
            params.n_memory, params.n_readout, reset_to_zero_state=True
        )
        self._identity_total = np.eye(self.dim_total, dtype=complex)
        self._fixed_unitary = self._build_random_circuit()

    def _rotation_z(self, angle: float) -> np.ndarray:
        return np.array(
            [
                [np.exp(-0.5j * angle), 0.0],
                [0.0, np.exp(0.5j * angle)],
            ],
            dtype=complex,
        )

    def _rotation_y(self, angle: float) -> np.ndarray:
        c = np.cos(0.5 * angle)
        s = np.sin(0.5 * angle)
        return np.array(
            [
                [c, -s],
                [s, c],
            ],
            dtype=complex,
        )

    def _random_su2_single_qubit(self, rng: np.random.Generator) -> np.ndarray:
        # Euler-angle construction for one random SU(2) gate.
        alpha = 2.0 * np.pi * rng.random()
        gamma = 2.0 * np.pi * rng.random()
        beta = 2.0 * np.arccos(np.sqrt(rng.random()))
        return (
            self._rotation_z(alpha) @ self._rotation_y(beta) @ self._rotation_z(gamma)
        )

    def _single_site_unitary(self, site: int, gate: np.ndarray) -> np.ndarray:
        # Expand a 2x2 gate in the Pauli basis, then lift it to the full register.
        coeff_i = 0.5 * np.trace(gate)
        coeff_x = 0.5 * np.trace(np.array([[0, 1], [1, 0]], dtype=complex) @ gate)
        coeff_y = 0.5 * np.trace(np.array([[0, -1j], [1j, 0]], dtype=complex) @ gate)
        coeff_z = 0.5 * np.trace(np.array([[1, 0], [0, -1]], dtype=complex) @ gate)
        return (
            coeff_i * self._identity_total
            + coeff_x * self._single(site, "X")
            + coeff_y * self._single(site, "Y")
            + coeff_z * self._single(site, "Z")
        )

    def _cnot_gate(self, control: int, target: int) -> np.ndarray:
        # CNOT = |0><0|_c ⊗ I_t + |1><1|_c ⊗ X_t, written in the Pauli basis.
        return 0.5 * (
            self._identity_total
            + self._single(control, "Z")
            + self._single(target, "X")
            - self._pair(control, "Z", target, "X")
        )

    def _build_random_circuit(self) -> np.ndarray:
        rng = np.random.default_rng(self.params.seed)
        U = self._identity_total.copy()

        for layer in range(self.params.depth):
            for site in range(self.n_total):
                gate = self._random_su2_single_qubit(rng)
                U = self._single_site_unitary(site, gate) @ U

            start = layer % 2
            # Alternate even and odd CNOT layers to produce a brickwork circuit
            # without trying to apply overlapping two-qubit gates at once.
            for control in range(start, self.n_total - 1, 2):
                target = control + 1
                U = self._cnot_gate(control, target) @ U

        return ensure_finite("random Pauli circuit unitary", U)

    def _build_unitary(self, u: float) -> np.ndarray:
        # The circuit itself is fixed; only the injected readout state depends on u.
        return self._fixed_unitary

    def _ghz_like_state(self, p: float, n_qubits: int) -> np.ndarray:
        if n_qubits == 1:
            return np.array([[np.sqrt(p)], [np.sqrt(1.0 - p)]], dtype=complex)

        state = np.zeros((2**n_qubits, 1), dtype=complex)
        state[0, 0] = np.sqrt(p)
        state[-1, 0] = np.sqrt(1.0 - p)
        return state

    def _input_reset_state(self, u: float) -> np.ndarray:
        u = float(u)
        p = float(
            np.clip(self.params.input_bias + self.params.input_scale * u, 0.0, 1.0)
        )
        left_qubits = self.params.input_qubit
        encoded_qubits = self.params.encoding_qubits
        right_qubits = self.n_readout - left_qubits - encoded_qubits

        left_state = computational_zero_state(left_qubits) if left_qubits > 0 else None
        encoded_state = self._ghz_like_state(p, encoded_qubits)
        right_state = (
            computational_zero_state(right_qubits) if right_qubits > 0 else None
        )

        # The encoded block is embedded into the full readout register by padding
        # with |0...0> blocks on both sides.
        state = encoded_state
        if left_state is not None:
            state = np.kron(left_state, state)
        if right_state is not None:
            state = np.kron(state, right_state)
        return state @ state.conj().T

    def kraus_operators(self, u: float) -> np.ndarray:
        u = float(u)
        cached = self._cache_get(self._kraus_cache, u)
        if cached is not None:
            return cached

        reset_state = self._input_reset_state(u)
        kraus = self._kraus_from_reset_state(
            u, reset_state, name="random_pauli_kraus_operators"
        )
        self._cache_set(self._kraus_cache, u, kraus)
        return kraus
