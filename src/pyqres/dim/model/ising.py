"""Ising-family dimension model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la

from ..linalg_utils import NumericalStabilityError, checked_matmul, ensure_finite
from .base import ReservoirBase


@dataclass
class IsingReservoirParameters:
    n_memory: int = 3
    n_readout: int = 1
    tau: float = 0.3
    gx_memory: float = 0.9
    gz_memory: float = 0.0
    jzz_memory: float = 1.0
    jxx_memory: float = 0.0
    jzz_next_nearest: float = 0.0
    gx_readout: float = 0.8
    gz_readout: float = 0.0
    kz_memory_readout: float = 0.7
    input_strength: float = 1.0
    input_axis: str = "Z"
    input_on_memory: bool = True
    input_site: int = 0
    input_sites: tuple[int, ...] | None = None
    input_strength_normalization: str = "none"
    periodic_memory_chain: bool = False
    reset_to_zero_state: bool = True

    # Hamiltonian used in the current Ising reservoir model:
    #
    #     H(u) = H0 + u H1
    #
    # where the static part H0 is
    #
    #     H0
    #       = sum_{i in memory}  (gx_memory  * X_i + gz_memory  * Z_i)
    #       + sum_{a in readout} (gx_readout * X_a + gz_readout * Z_a)
    #       + sum_{<i,j> in memory nn}      jzz_memory       * Z_i Z_j
    #       + sum_{<i,j> in memory nn}      jxx_memory       * X_i X_j
    #       + sum_{<<i,j>> in memory nnn}   jzz_next_nearest * Z_i Z_j
    #       + sum_{i in memory} sum_{a in readout} kz_memory_readout * Z_i Z_a
    #
    # and the input-dependent term is
    #
    #     H1 = input_strength * Z_{input_site}
    #
    # by default, or the corresponding normalized sum over input_sites when
    # multi-qubit scalar encoding is requested. The selected sites act either on
    # memory qubits or readout qubits depending on input_on_memory.
    #
    # Parameter meanings in H(u):
    #
    # - gx_memory:
    #     Strength of the transverse X field on each memory qubit.
    #
    # - gz_memory:
    #     Strength of the longitudinal Z field on each memory qubit.
    #
    # - gx_readout:
    #     Strength of the transverse X field on each readout qubit.
    #
    # - gz_readout:
    #     Strength of the longitudinal Z field on each readout qubit.
    #
    # - jzz_memory:
    #     Nearest-neighbor ZZ Ising coupling inside the memory chain.
    #
    # - jxx_memory:
    #     Nearest-neighbor XX coupling inside the memory chain.
    #     When nonzero, this deforms the pure transverse-field Ising structure.
    #
    # - jzz_next_nearest:
    #     Next-nearest-neighbor ZZ coupling inside the memory chain.
    #     This adds longer-range structure/frustration to the memory dynamics.
    #
    # - kz_memory_readout:
    #     ZZ coupling between every memory qubit and every readout qubit.
    #     This is the interaction that lets the reset readout subsystem affect
    #     the effective open dynamics of the memory subsystem.
    #
    # - input_strength:
    #     Overall scale of the input Hamiltonian H1.
    #     The scalar input u enters multiplicatively as u * input_strength.
    #
    # - input_on_memory:
    #     If True, the input Z drive is applied to a memory qubit.
    #     If False, it is applied to a readout qubit.
    #
    # - input_site:
    #     Index of the qubit on which the input drive acts when input_sites
    #     is not set.
    #
    # - input_sites:
    #     Optional list/tuple of qubit indices driven by the same scalar input.
    #     For example, [0, 1, 2] uses H1 proportional to Z0 + Z1 + Z2.
    #
    # - input_strength_normalization:
    #     Controls how the per-site input strength is scaled for multi-site
    #     encodings. none preserves the old per-site strength, sqrt keeps
    #     the Frobenius scale roughly comparable, and mean keeps the summed
    #     coefficient scale comparable.
    #
    # - periodic_memory_chain:
    #     Controls whether the memory-chain couplings use periodic boundary
    #     conditions (ring) or open boundary conditions (line).
    #
    # - tau:
    #     This does not change H(u) itself, but sets the one-step unitary
    #     evolution time through U(u) = exp(-1j * tau * H(u)).


class IsingReservoirModel(ReservoirBase):
    """Single-step Ising reservoir with input-dependent Hamiltonian.

    This is the simplest model in the package: one dense Hamiltonian H(u)
    generates one unitary step U(u) = exp(-i tau H(u)).
    """

    def __init__(self, params: IsingReservoirParameters):
        self.params = params
        self._initialize_common(
            params.n_memory, params.n_readout, params.reset_to_zero_state
        )
        # Precompute the static Hamiltonian and the H1 operator multiplied by the scalar input u.
        self._h0 = self._build_h0()
        self._h1 = self._build_h1()

    def _build_h0(self) -> np.ndarray:
        p = self.params
        H = np.zeros((self.dim_total, self.dim_total), dtype=complex)

        # The Hamiltonian is assembled in physically meaningful blocks: local memory
        # fields, local readout fields, memory-memory couplings, then memory-readout couplings.
        for i in range(p.n_memory):
            H += p.gx_memory * self._single(self._memory_site(i), "X")
            if abs(p.gz_memory) > 0:
                H += p.gz_memory * self._single(self._memory_site(i), "Z")

        for a in range(p.n_readout):
            H += p.gx_readout * self._single(self._readout_site(a), "X")
            if abs(p.gz_readout) > 0:
                H += p.gz_readout * self._single(self._readout_site(a), "Z")

        for i, j in self._memory_edges(p.periodic_memory_chain):
            H += p.jzz_memory * self._pair(
                self._memory_site(i), "Z", self._memory_site(j), "Z"
            )
            if abs(p.jxx_memory) > 0:
                H += p.jxx_memory * self._pair(
                    self._memory_site(i), "X", self._memory_site(j), "X"
                )

        if abs(p.jzz_next_nearest) > 0 and p.n_memory >= 3:
            for i, j in self._memory_next_nearest_edges(p.periodic_memory_chain):
                H += p.jzz_next_nearest * self._pair(
                    self._memory_site(i), "Z", self._memory_site(j), "Z"
                )

        for i in range(p.n_memory):
            for a in range(p.n_readout):
                H += p.kz_memory_readout * self._pair(
                    self._memory_site(i), "Z", self._readout_site(a), "Z"
                )

        # H is assembled term-by-term directly in the dense computational basis.
        return H

    def _build_h1(self) -> np.ndarray:
        p = self.params
        pauli = p.input_axis.upper()
        if pauli not in {"X", "Y", "Z"}:
            raise ValueError(f"Unsupported input_axis '{p.input_axis}'")
        sites = self._input_physical_sites(
            input_on_memory=p.input_on_memory,
            input_site=p.input_site,
            input_sites=p.input_sites,
        )
        scale = self._input_strength_prefactor(
            p.input_strength,
            len(sites),
            p.input_strength_normalization,
        )
        # The same scalar input u multiplies every selected Pauli generator.
        out = np.zeros((self.dim_total, self.dim_total), dtype=complex)
        for site in sites:
            out += scale * self._single(site, pauli)
        return out

    @property
    def h0(self) -> np.ndarray:
        return self._h0

    @property
    def h1(self) -> np.ndarray:
        return self._h1

    def _build_unitary(self, u: float) -> np.ndarray:
        # Form H(u) = H0 + u H1 and use a Hermitian eigendecomposition rather than a
        # general matrix exponential so numerical errors are easier to diagnose.
        H = self._h0 + u * self._h1
        ensure_finite("Hamiltonian", H)
        # Re-Hermitize after summation to suppress small floating-point asymmetries.
        H = 0.5 * (H + H.conj().T)
        hermitian_residual = np.linalg.norm(H - H.conj().T, ord="fro")
        if hermitian_residual > 1e-10:
            raise NumericalStabilityError(
                f"Hamiltonian lost Hermiticity for u={u}; residual={hermitian_residual:.3e}"
            )

        try:
            evals, evecs = la.eigh(H, check_finite=True)
        except Exception as exc:
            raise NumericalStabilityError(
                f"Hermitian eigendecomposition failed for u={u}, tau={self.params.tau}; "
                f"shape={H.shape}, max_abs={np.max(np.abs(H)):.3e}, norm_fro={np.linalg.norm(H, ord='fro'):.3e}"
            ) from exc
        ensure_finite("Hamiltonian eigenvalues", evals)
        ensure_finite("Hamiltonian eigenvectors", evecs)
        phases = np.exp(-1j * self.params.tau * evals)
        ensure_finite("unitary phases", phases)
        phased_evecs = evecs * phases[np.newaxis, :]
        # Reconstruct exp(-i tau H) from the Hermitian eigendecomposition.
        return checked_matmul(
            "unitary eigendecomposition reconstruction", phased_evecs, evecs.conj().T
        )
