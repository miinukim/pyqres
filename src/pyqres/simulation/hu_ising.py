"""Driven Ising reservoir with partial readout reset.

This module implements the numerical Hamiltonian ansatz introduced in
F. Hu et al., Nature Communications 15, 7491 (2024):

    H(u) = H0 + u H1
    H0 = sum_(i<j) J_ij Z_i Z_j + sum_i eta_i^x X_i
    H1 = sum_i eta_i^z Z_i.

The input-dependent evolution is evaluated with one first-order digital step,
``exp(-i tau H0) exp(-i tau u H1)``. This keeps a fixed static propagator and
is the scalable counterpart of the paper's Trotterized gate construction.
After every step the readout marginal is measured and reset to zero; only the
memory marginal persists. Readout features can be either exact expectations or
finite-shot local-Pauli classical-shadow estimates.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import scipy.linalg as la

from pyqres.prethermal_shadow.dynamics import pauli_string, project_density
from pyqres.prethermal_shadow.shadows import (
    exact_readout_expectations,
    generate_pauli_labels,
    partial_shadow_feature_names,
    sample_partial_shadow_features,
)


@dataclass(frozen=True)
class IsingReservoirConfig:
    """Parameters for the numerical Ising reservoir.

    The defaults reproduce the disorder distributions quoted for the paper's
    numerical channel-equalization study, while allowing the memory/readout
    partition to be enlarged for resource-matched comparisons.
    """

    n_memory: int = 2
    n_readout: int = 4
    tau: float = 1.0
    j_max: float = 1.0
    eta_x: float = 2.0
    eta_x_rms: float = 2.0
    eta_z: float = 0.5
    eta_z_rms: float = 0.5
    pauli_k: int = 2
    include_bias: bool = True
    seed: int = 11
    exact_expectations: bool = True
    shots: int = 1024
    measurement_type: str = "projective"
    weak_strength: float = 1.0
    shadow_seed: int = 2
    return_raw_shots: bool = False

    def __post_init__(self) -> None:
        if self.n_memory < 1 or self.n_readout < 1:
            raise ValueError("n_memory and n_readout must be positive")
        if self.tau <= 0.0:
            raise ValueError("tau must be positive")
        if self.j_max < 0.0 or self.eta_x_rms < 0.0 or self.eta_z_rms < 0.0:
            raise ValueError("coupling and disorder scales must be non-negative")
        if self.pauli_k < 1 or self.pauli_k > self.n_readout:
            raise ValueError("pauli_k must lie between 1 and n_readout")
        if self.shots < 1:
            raise ValueError("shots must be positive")
        if self.measurement_type.lower() not in {"projective", "weak"}:
            raise ValueError("measurement_type must be projective or weak")
        if not 0.0 < self.weak_strength <= 1.0:
            raise ValueError("weak_strength must lie in (0, 1]")


class IsingReservoir:
    """Dense-state streaming adapter for the driven Ising architecture.

    Qubits are ordered memory first and readout second. The fixed readout reset
    state is ``|0...0>``. Readout features contain all readout-local Pauli
    strings through ``pauli_k`` and are exact expectations or finite-shot
    classical-shadow estimates according to ``config.exact_expectations``.
    """

    def __init__(self, config: IsingReservoirConfig):
        self.config = config
        self.n_memory = int(config.n_memory)
        self.n_readout = int(config.n_readout)
        self.n_qubits = self.n_memory + self.n_readout
        self.dim_memory = 2**self.n_memory
        self.dim_readout = 2**self.n_readout
        self.dim_total = self.dim_memory * self.dim_readout

        rng = np.random.default_rng(config.seed)
        self.couplings = np.zeros((self.n_qubits, self.n_qubits), dtype=float)
        for i, j in combinations(range(self.n_qubits), 2):
            self.couplings[i, j] = self.couplings[j, i] = rng.uniform(0.0, config.j_max)
        self.eta_x_values = config.eta_x + rng.normal(
            0.0, config.eta_x_rms, size=self.n_qubits
        )
        self.eta_z_values = config.eta_z + rng.normal(
            0.0, config.eta_z_rms, size=self.n_qubits
        )

        self.h0 = self._build_h0()
        self.h1_diagonal = self._build_h1_diagonal()
        self.static_unitary = la.expm(-1j * config.tau * self.h0)

        # With memory-first tensor ordering, these are the columns whose input
        # readout index is zero. Reshaping exposes the output readout index.
        reset_columns = np.arange(self.dim_memory, dtype=int) * self.dim_readout
        self._reset_isometry = self.static_unitary[:, reset_columns].reshape(
            self.dim_memory, self.dim_readout, self.dim_memory
        )
        self._input_diagonal = self.h1_diagonal[reset_columns]
        self._labels = generate_pauli_labels(config.n_readout, config.pauli_k)
        self._feature_names = partial_shadow_feature_names(
            config.n_readout,
            config.pauli_k,
            include_bias=config.include_bias,
        )
        self.last_raw_shots: dict[str, np.ndarray] | None = None
        self.last_readout_state: np.ndarray | None = None
        self.reset_state()

    def _build_h0(self) -> np.ndarray:
        hamiltonian = np.zeros((self.dim_total, self.dim_total), dtype=complex)
        for site, strength in enumerate(self.eta_x_values):
            hamiltonian += float(strength) * pauli_string(self.n_qubits, [(site, "X")])
        for i, j in combinations(range(self.n_qubits), 2):
            hamiltonian += self.couplings[i, j] * pauli_string(
                self.n_qubits, [(i, "Z"), (j, "Z")]
            )
        return 0.5 * (hamiltonian + hamiltonian.conj().T)

    def _build_h1_diagonal(self) -> np.ndarray:
        basis = np.arange(self.dim_total, dtype=np.uint64)
        diagonal = np.zeros(self.dim_total, dtype=float)
        # pauli_string uses big-endian Kronecker ordering: qubit zero is the
        # most significant computational-basis bit.
        for site, strength in enumerate(self.eta_z_values):
            shift = self.n_qubits - site - 1
            z_eigenvalue = 1.0 - 2.0 * ((basis >> shift) & 1).astype(float)
            diagonal += float(strength) * z_eigenvalue
        return diagonal

    def reset_state(self) -> None:
        """Reset the persistent memory and replayable shadow RNG."""

        self.rho_memory = np.zeros((self.dim_memory, self.dim_memory), dtype=complex)
        self.rho_memory[0, 0] = 1.0
        self._shadow_rng = np.random.default_rng(self.config.shadow_seed)
        self.last_raw_shots = None
        self.last_readout_state = None

    def reset(self) -> None:
        """Alias compatible with task-side streaming protocols."""

        self.reset_state()

    def step(self, u: float) -> np.ndarray:
        """Advance one input, emit readout features, and reset readout."""

        phase = np.exp(-1j * self.config.tau * float(u) * self._input_diagonal)
        driven_memory = (
            phase[:, np.newaxis] * self.rho_memory * phase.conj()[np.newaxis, :]
        )
        isometry = self._reset_isometry
        rho_readout = np.einsum(
            "mra,ab,msb->rs",
            isometry,
            driven_memory,
            isometry.conj(),
            optimize=True,
        )
        rho_memory = np.einsum(
            "mra,ab,nrb->mn",
            isometry,
            driven_memory,
            isometry.conj(),
            optimize=True,
        )
        rho_memory = 0.5 * (rho_memory + rho_memory.conj().T)
        trace = np.trace(rho_memory)
        if abs(trace) <= 1e-14:
            raise FloatingPointError("Ising reservoir memory state lost normalization")
        self.rho_memory = rho_memory / trace
        rho_readout = project_density(rho_readout)
        self.last_readout_state = rho_readout.copy()
        if self.config.exact_expectations:
            self.last_raw_shots = None
            return exact_readout_expectations(
                rho_readout,
                self._labels,
                include_bias=self.config.include_bias,
            )
        result = sample_partial_shadow_features(
            rho_readout,
            self._labels,
            shots=self.config.shots,
            measurement_type=self.config.measurement_type,
            weak_strength=self.config.weak_strength,
            include_bias=self.config.include_bias,
            rng=self._shadow_rng,
            return_raw=self.config.return_raw_shots,
        )
        if isinstance(result, tuple):
            features, raw = result
            self.last_raw_shots = raw
            return np.asarray(features, dtype=float)
        self.last_raw_shots = None
        return np.asarray(result, dtype=float)

    def run_stream(self, inputs: Sequence[float]) -> np.ndarray:
        """Reset and collect one feature row for every scalar input."""

        self.reset_state()
        values = [self.step(float(value)) for value in inputs]
        if not values:
            return np.empty((0, len(self._feature_names)), dtype=float)
        features = np.vstack(values)
        if not np.isfinite(features).all():
            raise FloatingPointError("Ising reservoir produced non-finite features")
        return features

    def get_feature_names(self) -> list[str]:
        return list(self._feature_names)


# Backward-compatible aliases for code written against releases before the
# neutral model naming was introduced.
HuIsingConfig = IsingReservoirConfig
HuIsingReservoir = IsingReservoir

__all__ = [
    "HuIsingConfig",
    "HuIsingReservoir",
    "IsingReservoir",
    "IsingReservoirConfig",
]
