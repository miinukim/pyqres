"""Number-conserving SYK dimension model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import scipy.linalg as la

from ..linalg_utils import NumericalStabilityError, ensure_finite
from ..pauli import computational_zero_state
from .base import ReservoirBase


def _sigma_minus() -> np.ndarray:
    return np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)


def _sigma_z() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def _single_qubit_identity() -> np.ndarray:
    return np.eye(2, dtype=complex)


def _jordan_wigner_annihilation(n_sites: int, site: int) -> np.ndarray:
    ops = []
    for idx in range(n_sites):
        if idx < site:
            ops.append(_sigma_z())
        elif idx == site:
            ops.append(_sigma_minus())
        else:
            ops.append(_single_qubit_identity())
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def _complex_normal_matrix(
    rng: np.random.Generator, shape: tuple[int, ...], scale: float
) -> np.ndarray:
    return scale * (rng.normal(size=shape) + 1j * rng.normal(size=shape))


@dataclass
class SYKReservoirParameters:
    n_memory: int = 7
    n_readout: int = 1
    tau: float = 1.0
    j4_strength: float = 1.0
    kappa2_strength: float = 0.0
    seed: int = 1234
    input_qubit: int = 0
    input_bias: float = 0.0
    input_scale: float = 1.0
    input_clip_eps: float = 1.0e-9
    normalize_syk4_by_spectral_norm: bool = False
    normalize_syk2_by_spectral_norm: bool = False
    reset_to_zero_state: bool = True


class SYKReservoirModel(ReservoirBase):
    """Number-conserving fermionic SYK reservoir with one encoded readout qubit.

    The scalar input is encoded on one chosen readout qubit. Any additional
    readout qubits are reset to |0>, so increasing n_readout enlarges the
    traced-out environment without changing the one-scalar input protocol.
    """

    def __init__(self, params: SYKReservoirParameters):
        if params.n_readout < 1:
            raise ValueError("SYKReservoirModel requires n_readout >= 1.")
        if not (0 <= params.input_qubit < params.n_readout):
            raise ValueError(
                f"input_qubit={params.input_qubit} must lie in [0, {params.n_readout - 1}]"
            )
        self.params = params
        self._initialize_common(
            params.n_memory, params.n_readout, params.reset_to_zero_state
        )
        self._rng = np.random.default_rng(params.seed)
        self._annihilation_ops = [
            self._build_annihilation(site) for site in range(self.n_total)
        ]
        self._creation_ops = [op.conj().T for op in self._annihilation_ops]
        self._number_ops_memory = self._build_memory_number_ops()
        self._syk4_h = self._build_syk4_hamiltonian()
        self._syk2_h = self._build_syk2_hamiltonian()
        if params.normalize_syk4_by_spectral_norm and np.any(self._syk4_h):
            norm4 = float(la.svdvals(self._syk4_h)[0])
            if norm4 > 0:
                self._syk4_h = self._syk4_h / norm4
        if params.normalize_syk2_by_spectral_norm and np.any(self._syk2_h):
            norm2 = float(la.svdvals(self._syk2_h)[0])
            if norm2 > 0:
                self._syk2_h = self._syk2_h / norm2
        self._hamiltonian = ensure_finite(
            "SYK Hamiltonian", self._syk4_h + self._syk2_h
        )
        self._hamiltonian = 0.5 * (self._hamiltonian + self._hamiltonian.conj().T)
        evals, evecs = la.eigh(self._hamiltonian, check_finite=True)
        self._hamiltonian_evals = ensure_finite("SYK Hamiltonian eigenvalues", evals)
        self._hamiltonian_evecs = ensure_finite("SYK Hamiltonian eigenvectors", evecs)
        phases = np.exp(-1j * self.params.tau * self._hamiltonian_evals)
        self._unitary = ensure_finite(
            "SYK unitary",
            (self._hamiltonian_evecs * phases[np.newaxis, :])
            @ self._hamiltonian_evecs.conj().T,
        )

    def _build_annihilation(self, site: int) -> np.ndarray:
        return _jordan_wigner_annihilation(self.n_total, site)

    def _build_memory_number_ops(self) -> list[np.ndarray]:
        out = []
        for site in range(self.n_memory):
            ann = _jordan_wigner_annihilation(self.n_memory, site)
            out.append(ann.conj().T @ ann)
        return out

    def _build_syk4_hamiltonian(self) -> np.ndarray:
        n_sites = self.n_total
        pair_indices = list(combinations(range(n_sites), 2))
        if not pair_indices or self.params.j4_strength == 0:
            return np.zeros((self.dim_total, self.dim_total), dtype=complex)

        variance = (self.params.j4_strength**2) / float(n_sites**3)
        scale = np.sqrt(variance / 2.0)
        raw = _complex_normal_matrix(
            self._rng, (len(pair_indices), len(pair_indices)), scale=scale
        )
        coupling_matrix = 0.5 * (raw + raw.conj().T)

        pair_create = [
            self._creation_ops[i] @ self._creation_ops[j] for i, j in pair_indices
        ]
        pair_annihilate = [
            self._annihilation_ops[k] @ self._annihilation_ops[l]
            for k, l in pair_indices
        ]
        h4 = np.zeros((self.dim_total, self.dim_total), dtype=complex)
        for a, create_op in enumerate(pair_create):
            for b, annihilate_op in enumerate(pair_annihilate):
                # This is the dense operator form of the complex SYK4 interaction
                # c_i^\dagger c_j^\dagger c_k c_l in a number-conserving basis.
                coeff = coupling_matrix[a, b]
                if abs(coeff) > 0:
                    h4 += coeff * (create_op @ annihilate_op)
        return 0.5 * (h4 + h4.conj().T)

    def _build_syk2_hamiltonian(self) -> np.ndarray:
        n_sites = self.n_total
        if self.params.kappa2_strength == 0:
            return np.zeros((self.dim_total, self.dim_total), dtype=complex)
        variance = (self.params.kappa2_strength**2) / float(2 * n_sites)
        coupling = np.zeros((n_sites, n_sites), dtype=complex)
        diag_scale = np.sqrt(variance)
        for i in range(n_sites):
            coupling[i, i] = self._rng.normal(scale=diag_scale)
        offdiag_scale = np.sqrt(variance / 2.0)
        for i in range(n_sites):
            for j in range(i + 1, n_sites):
                value = offdiag_scale * (self._rng.normal() + 1j * self._rng.normal())
                coupling[i, j] = value
                coupling[j, i] = np.conjugate(value)

        h2 = np.zeros((self.dim_total, self.dim_total), dtype=complex)
        for i in range(n_sites):
            for j in range(n_sites):
                coeff = coupling[i, j]
                if abs(coeff) > 0:
                    h2 += coeff * (self._creation_ops[i] @ self._annihilation_ops[j])
        return 0.5 * (h2 + h2.conj().T)

    @property
    def hamiltonian(self) -> np.ndarray:
        return self._hamiltonian.copy()

    @property
    def syk4_hamiltonian(self) -> np.ndarray:
        return self._syk4_h.copy()

    @property
    def syk2_hamiltonian(self) -> np.ndarray:
        return self._syk2_h.copy()

    def _encoded_probability(self, u: float) -> float:
        p = float(self.params.input_bias + self.params.input_scale * u)
        return float(
            np.clip(p, self.params.input_clip_eps, 1.0 - self.params.input_clip_eps)
        )

    def input_state_vector(self, u: float) -> np.ndarray:
        p = self._encoded_probability(u)
        return np.array([np.sqrt(1.0 - p), np.sqrt(p)], dtype=complex)

    def _input_reset_state(self, u: float) -> np.ndarray:
        encoded_state = self.input_state_vector(u).reshape(2, 1)
        left_qubits = self.params.input_qubit
        right_qubits = self.n_readout - left_qubits - 1

        # Only one readout qubit carries the scalar input; any extra readout
        # qubits simply enlarge the environment and are reset to |0>.
        state = encoded_state
        if left_qubits > 0:
            state = np.kron(computational_zero_state(left_qubits), state)
        if right_qubits > 0:
            state = np.kron(state, computational_zero_state(right_qubits))
        return state @ state.conj().T

    def _build_unitary(self, u: float) -> np.ndarray:
        return self._unitary

    def kraus_operators(self, u: float) -> np.ndarray:
        u = float(u)
        cached = self._cache_get(self._kraus_cache, u)
        if cached is not None:
            return cached
        reset_state = self._input_reset_state(u)
        kraus = self._kraus_from_reset_state(u, reset_state, name="SYK Kraus operators")
        self._cache_set(self._kraus_cache, u, kraus)
        return kraus

    def parse_memory_observable(self, spec: str) -> np.ndarray:
        cleaned = spec.replace(" ", "")
        if not cleaned:
            raise ValueError("Observable spec must be non-empty")
        if cleaned[0] in {"N", "n"}:
            factors = cleaned.split("*")
            out = np.eye(self.dim_memory, dtype=complex)
            for token in factors:
                if token[0] not in {"N", "n"}:
                    raise ValueError(
                        f"Unsupported SYK number observable token '{token}'"
                    )
                site = int(token[1:])
                if not (0 <= site < self.n_memory):
                    raise ValueError(
                        f"Observable token '{token}' is out of range for n_memory={self.n_memory}"
                    )
                # Number observables are multiplied directly, so strings like
                # N0*N1 become products of local occupation operators.
                out = out @ self._number_ops_memory[site]
            return out
        return super().parse_memory_observable(spec)

    def default_memory_observable_specs(
        self,
        preset: str = "occupation",
        custom_specs: Sequence[str] | None = None,
    ) -> list[str]:
        preset_key = preset.lower()
        if preset_key in {"occupation", "occupations", "number"}:
            obs_specs = [f"N{i}" for i in range(self.n_memory)]
        elif preset_key == "occupation_pairs":
            obs_specs = [f"N{i}*N{j}" for i, j in combinations(range(self.n_memory), 2)]
        elif preset_key == "occupation_rich":
            obs_specs = [f"N{i}" for i in range(self.n_memory)] + [
                f"N{i}*N{j}" for i, j in combinations(range(self.n_memory), 2)
            ]
        else:
            obs_specs = super().default_memory_observable_specs(
                preset=preset, custom_specs=None
            )
        if custom_specs:
            obs_specs.extend(custom_specs)
        return list(dict.fromkeys(obs_specs))

    def particle_number_sector_indices(self, total_particles: int) -> np.ndarray:
        if not (0 <= total_particles <= self.n_total):
            raise ValueError(
                f"total_particles={total_particles} must lie in [0, {self.n_total}]"
            )
        return np.array(
            [
                idx
                for idx in range(self.dim_total)
                if idx.bit_count() == total_particles
            ],
            dtype=int,
        )

    def sector_hamiltonian(self, total_particles: int) -> np.ndarray:
        indices = self.particle_number_sector_indices(total_particles)
        if indices.size == 0:
            raise NumericalStabilityError(
                f"No basis states in particle-number sector Np={total_particles}"
            )
        return self._hamiltonian[np.ix_(indices, indices)]

    def mean_level_spacing_ratio(
        self, total_particles: int | None = None, central_fraction: float = 0.5
    ) -> float:
        if total_particles is None:
            total_particles = self.n_total // 2
        sector_h = self.sector_hamiltonian(total_particles)
        evals = np.sort(la.eigvalsh(sector_h, check_finite=True))
        if evals.size < 3:
            raise NumericalStabilityError(
                "Need at least 3 eigenvalues to compute spacing ratios."
            )
        fraction = float(np.clip(central_fraction, 0.0, 1.0))
        if 0.0 < fraction < 1.0:
            keep = max(3, round(fraction * evals.size))
            start = max(0, (evals.size - keep) // 2)
            evals = evals[start : start + keep]
        spacings = np.diff(evals)
        ratios = np.minimum(spacings[:-1] / spacings[1:], spacings[1:] / spacings[:-1])
        ratios = ratios[np.isfinite(ratios)]
        if ratios.size == 0:
            raise NumericalStabilityError("No finite spacing ratios were produced.")
        return float(np.mean(ratios))
