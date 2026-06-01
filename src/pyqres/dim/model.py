"""Reservoir models and PTM construction.

This module contains two closely related layers:

1. a common reservoir-channel interface that turns a joint memory/readout
   unitary into an effective channel on the memory subsystem
2. concrete Hamiltonian families that provide the unitary for a given scalar
   input u

The mathematical object consumed by the analysis code is the memory-channel PTM
(Pauli transfer matrix). Everything in this module is organized around building
that PTM from dense operators in the computational basis.
"""

from collections import OrderedDict
from dataclasses import dataclass
from itertools import combinations, product
from typing import List, Sequence

import numpy as np
import scipy.linalg as la

from .linalg_utils import (
    NumericalStabilityError,
    checked_matmul,
    derivative_from_samples,
    ensure_finite,
)
from .pauli import (
    computational_zero_density,
    pauli_basis_matrices,
    pauli_string,
    single_site_pauli,
    two_site_pauli,
)


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


def _complex_normal_matrix(rng: np.random.Generator, shape: tuple[int, ...], scale: float) -> np.ndarray:
    return scale * (rng.normal(size=shape) + 1j * rng.normal(size=shape))


class ReservoirBase:
    """Common reservoir-channel interface used by the PTM/Volterra analysis code.

    A concrete subclass only needs to define how the joint unitary U(u) is constructed. This base class then:

    - converts that joint unitary into Kraus operators on the memory subsystem
    - applies the resulting memory channel to operators
    - projects the channel into the Pauli basis to form the PTM
    - computes a zero-input fixed point used by the Volterra analysis
    """

    def _initialize_common(
        self,
        n_memory: int,
        n_readout: int,
        reset_to_zero_state: bool,
        cache_max_entries: int | None = 1024,
    ) -> None:
        # memory is the subsystem whose effective open dynamics we analyze.
        # readout is reset after each step, which induces the effective memory channel.
        self.n_memory = n_memory
        self.n_readout = n_readout
        self.n_total = n_memory + n_readout
        self.dim_memory = 2**n_memory
        self.dim_readout = 2**n_readout
        self.dim_total = 2**self.n_total
        # The PTM is expressed in the full Pauli basis of the memory subsystem.
        self.memory_basis = pauli_basis_matrices(n_memory)
        self._memory_basis_stack = np.stack(self.memory_basis, axis=0)
        self.reset_state = (
            computational_zero_density(n_readout)
            if reset_to_zero_state
            else np.eye(self.dim_readout, dtype=complex) / self.dim_readout
        )
        # Cache expensive dense objects by input value u because sweeps revisit the same samples.
        self._cache_max_entries = cache_max_entries
        self._unitary_cache: OrderedDict[float, np.ndarray] = OrderedDict()
        self._kraus_cache: OrderedDict[float, np.ndarray] = OrderedDict()
        self._ptm_cache: OrderedDict[float, np.ndarray] = OrderedDict()
        self._fixed_point_cache: np.ndarray | None = None

    def _cache_get(self, cache: OrderedDict[float, np.ndarray], key: float) -> np.ndarray | None:
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
        return cached

    def _cache_set(self, cache: OrderedDict[float, np.ndarray], key: float, value: np.ndarray) -> None:
        cache[key] = value
        cache.move_to_end(key)
        if self._cache_max_entries is not None:
            if self._cache_max_entries <= 0:
                cache.clear()
                return
            while len(cache) > self._cache_max_entries:
                cache.popitem(last=False)

    def clear_caches(self) -> None:
        """Release cached dense unitary/Kraus/PTM arrays held by this model."""

        self._unitary_cache.clear()
        self._kraus_cache.clear()
        self._ptm_cache.clear()

    def _memory_site(self, idx: int) -> int:
        return idx

    def _readout_site(self, idx: int) -> int:
        return self.n_memory + idx

    def _input_physical_sites(
        self,
        *,
        input_on_memory: bool = True,
        input_site: int,
        input_sites: Sequence[int] | None,
    ) -> tuple[int, ...]:
        """
        Convert logical input-site indices into joint-system site indices.
        """

        logical_sites = tuple(int(site) for site in input_sites) if input_sites is not None else (int(input_site),)
        if not logical_sites:
            raise ValueError("At least one input site is required")

        physical_sites: list[int] = []
        for site in logical_sites:
            if input_on_memory:
                if not (0 <= site < self.n_memory):
                    raise ValueError(f"Input memory site {site} is out of range for n_memory={self.n_memory}")
                physical_sites.append(self._memory_site(site))
            else:
                if not (0 <= site < self.n_readout):
                    raise ValueError(f"Input readout site {site} is out of range for n_readout={self.n_readout}")
                physical_sites.append(self._readout_site(site))
        return tuple(physical_sites)

    def _input_strength_prefactor(self, strength: float, n_sites: int, normalization: str) -> float:
        """Return the per-site drive scale for a multi-qubit scalar encoding."""

        mode = normalization.lower()
        if mode in {"none", "sum"}:
            return float(strength)
        if mode in {"sqrt", "frobenius"}:
            return float(strength) / float(np.sqrt(n_sites))
        if mode in {"mean", "average"}:
            return float(strength) / float(n_sites)
        raise ValueError(
            "input_strength_normalization must be one of "
            "{'none', 'sum', 'sqrt', 'frobenius', 'mean', 'average'}"
        )

    def _single(self, site: int, pauli: str) -> np.ndarray:
        return single_site_pauli(self.n_total, site, pauli)

    def _pair(self, site_a: int, pauli_a: str, site_b: int, pauli_b: str) -> np.ndarray:
        return two_site_pauli(self.n_total, site_a, pauli_a, site_b, pauli_b)

    def _memory_edges(self, periodic: bool) -> list[tuple[int, int]]:
        # Helper for nearest-neighbor couplings along the memory chain.
        edges = [(i, i + 1) for i in range(self.n_memory - 1)]
        if periodic and self.n_memory > 2:
            edges.append((self.n_memory - 1, 0))
        return edges

    def _memory_next_nearest_edges(self, periodic: bool) -> list[tuple[int, int]]:
        # Helper for next-nearest-neighbor couplings used in the non-integrable deformations.
        edges = [(i, i + 2) for i in range(self.n_memory - 2)]
        if periodic and self.n_memory > 3:
            edges.extend([(self.n_memory - 2, 0), (self.n_memory - 1, 1)])
        return edges

    def _build_unitary(self, u: float) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def unitary(self, u: float) -> np.ndarray:
        # Cache by scalar input u because PTM differentiation samples the same values repeatedly.
        u = float(u)
        cached = self._cache_get(self._unitary_cache, u)
        if cached is not None:
            return cached
        U = ensure_finite("unitary", self._build_unitary(u))
        self._cache_set(self._unitary_cache, u, U)
        return U

    def kraus_operators(self, u: float) -> np.ndarray:
        u = float(u)
        cached = self._cache_get(self._kraus_cache, u)
        if cached is not None:
            return cached
        U = self.unitary(u)
        # View the joint unitary as memory-out / readout-in / memory-in / readout-out indices.
        U4 = U.reshape(self.dim_memory, self.dim_readout, self.dim_memory, self.dim_readout)
        evals, evecs = la.eigh(self.reset_state, check_finite=True)
        active = evals > 1e-15
        if not np.any(active):
            raise NumericalStabilityError("Reset state has no positive eigenvalues")

        blocks = []
        for weight, psi in zip(evals[active], evecs[:, active].T, strict=False):
            # If the readout reset state is rho_R = sum_j w_j |psi_j><psi_j|, then
            # the effective memory channel is obtained by contracting U against
            # each populated readout eigenvector |psi_j>. The resulting blocks are
            # stacked into a conventional Kraus list over the memory subsystem.
            contracted = np.einsum("arbi,i->arb", U4, psi, optimize=True)
            blocks.append(np.sqrt(weight) * np.transpose(contracted, (1, 0, 2)))
        kraus = ensure_finite(f"kraus_operators(u={u})", np.concatenate(blocks, axis=0))
        self._cache_set(self._kraus_cache, u, kraus)
        return kraus

    def channel(self, u: float, op_memory: np.ndarray) -> np.ndarray:
        ensure_finite("memory operator", op_memory)
        kraus = self.kraus_operators(u)
        # Apply the CPTP map Phi_u(X) = sum_k K_k X K_k^\dagger on the memory subsystem.
        out = np.einsum("kab,bc,kdc->ad", kraus, op_memory, kraus.conj(), optimize=True)
        return ensure_finite("channel output", out)

    def channel_adjoint(self, u: float, observable_memory: np.ndarray) -> np.ndarray:
        ensure_finite("memory observable", observable_memory)
        kraus = self.kraus_operators(u)
        out = np.einsum("kba,bc,kcd->ad", kraus.conj(), observable_memory, kraus, optimize=True)
        return ensure_finite("adjoint channel output", out)

    def channel_derivative_adjoint(
        self,
        order: int,
        observable_memory: np.ndarray,
        u0: float = 0.0,
        fd_step: float = 5e-3,
        radius: int | None = None,
    ) -> np.ndarray:
        if order < 0:
            raise ValueError(f"Derivative order must be non-negative, got {order}.")
        if order == 0:
            return self.channel_adjoint(u0, observable_memory)
        radius_value = radius if radius is not None else max(2, order + 1)
        points = list(range(-radius_value, radius_value + 1))
        # The implementation is matrix-free with respect to the PTM: it samples
        # the adjoint channel directly on the current observable and then applies
        # a scalar finite-difference stencil entrywise.
        samples = [
            self.channel_adjoint(float(u0 + p * fd_step), observable_memory)
            for p in points
        ]
        return ensure_finite(
            f"adjoint channel derivative order {order}",
            derivative_from_samples(samples, fd_step, order, points),
        )

    def ptm(self, u: float) -> np.ndarray:
        u = float(u)
        cached = self._cache_get(self._ptm_cache, u)
        if cached is not None:
            return cached
        kraus = self.kraus_operators(u)
        # First push every Pauli basis element through the channel in one batched contraction.
        # outputs[n] is Phi_u(P_n), still represented as a dense memory operator.
        outputs = np.einsum(
            "kab,nbc,kdc->nad",
            kraus,
            self._memory_basis_stack,
            kraus.conj(),
            optimize=True,
        )
        # Then project the outputs back onto the Pauli basis to obtain the PTM entries
        # T_{mn} = tr(P_m Phi_u(P_n)) / dim_memory.
        T = np.einsum(
            "mab,nab->mn",
            self._memory_basis_stack.conj(),
            outputs,
            optimize=True,
        ) / self.dim_memory
        T = ensure_finite(f"PTM(u={u})", T)
        self._cache_set(self._ptm_cache, u, T)
        return T

    def readout_matrix(self, observables: Sequence[np.ndarray]) -> np.ndarray:
        dim = self.dim_memory
        traceless_basis = self.memory_basis[1:]
        R = np.zeros((len(observables), len(traceless_basis)), dtype=complex)
        for j, M in enumerate(observables):
            for mu, P in enumerate(traceless_basis):
                # R converts traceless PTM coordinates into expectation values of chosen observables.
                R[j, mu] = np.trace(M @ P) / dim
        return R

    def parse_memory_observable(self, spec: str) -> np.ndarray:
        # Accepted syntax is a product such as Z0*X2 acting only on memory sites.
        cleaned = spec.replace(" ", "")
        if not cleaned:
            raise ValueError("Observable spec must be non-empty")
        factors = []
        for token in cleaned.split("*"):
            pauli = token[0].upper()
            if pauli not in {"X", "Y", "Z"}:
                raise ValueError(f"Unsupported Pauli observable token '{token}'")
            try:
                site = int(token[1:])
            except ValueError as exc:
                raise ValueError(f"Observable token '{token}' must have an integer site index") from exc
            if not (0 <= site < self.n_memory):
                raise ValueError(f"Observable token '{token}' is out of range for n_memory={self.n_memory}")
            factors.append((site, pauli))
        return pauli_string(self.n_memory, tuple(sorted(factors)))

    def _single_site_specs(self, paulis: Sequence[str]) -> List[str]:
        return [f"{pauli}{site}" for pauli in paulis for site in range(self.n_memory)]

    def _pair_specs(self, paulis_left: Sequence[str], paulis_right: Sequence[str]) -> List[str]:
        specs: List[str] = []
        for left_site, right_site in combinations(range(self.n_memory), 2):
            for left_pauli, right_pauli in product(paulis_left, paulis_right):
                specs.append(f"{left_pauli}{left_site}*{right_pauli}{right_site}")
        return specs

    def _nearest_neighbor_pair_specs(self, paulis_left: Sequence[str], paulis_right: Sequence[str]) -> List[str]:
        specs: List[str] = []
        for left_site in range(self.n_memory - 1):
            right_site = left_site + 1
            for left_pauli, right_pauli in product(paulis_left, paulis_right):
                specs.append(f"{left_pauli}{left_site}*{right_pauli}{right_site}")
        return specs

    def default_memory_observable_specs(
        self,
        preset: str = "z",
        custom_specs: Sequence[str] | None = None,
    ) -> List[str]:
        # These presets are convenience libraries for common readout choices used
        # in the experiments. The return value is still just a list of strings so
        # callers can inspect or augment it before materializing dense operators.
        preset_key = preset.lower()
        obs_specs: List[str]
        if preset_key == "z":
            obs_specs = [f"Z{i}" for i in range(self.n_memory)]
        elif preset_key == "x":
            obs_specs = [f"X{i}" for i in range(self.n_memory)]
        elif preset_key == "y":
            obs_specs = [f"Y{i}" for i in range(self.n_memory)]
        elif preset_key == "xy":
            obs_specs = self._single_site_specs(("X", "Y"))
        elif preset_key == "zx":
            obs_specs = [f"Z{i}" for i in range(self.n_memory)] + [f"X{i}" for i in range(self.n_memory)]
        elif preset_key == "xyz":
            obs_specs = self._single_site_specs(("X", "Y", "Z"))
        elif preset_key == "zz_pairs":
            obs_specs = self._pair_specs(("Z",), ("Z",))
        elif preset_key == "xx_pairs":
            obs_specs = self._pair_specs(("X",), ("X",))
        elif preset_key == "nn_pairs":
            obs_specs = self._nearest_neighbor_pair_specs(("X", "Y", "Z"), ("X", "Y", "Z"))
        elif preset_key == "pair_xyz":
            obs_specs = self._pair_specs(("X", "Y", "Z"), ("X", "Y", "Z"))
        elif preset_key == "rich":
            obs_specs = self._single_site_specs(("X", "Y", "Z")) + self._pair_specs(
                ("X", "Y", "Z"),
                ("X", "Y", "Z"),
            )
        elif preset_key == "custom":
            obs_specs = []
        else:
            raise ValueError(f"Unsupported observable preset '{preset}'")

        if custom_specs:
            obs_specs.extend(custom_specs)

        return list(dict.fromkeys(obs_specs))

    def default_memory_observables(
        self,
        preset: str = "z",
        custom_specs: Sequence[str] | None = None,
    ) -> List[np.ndarray]:
        deduped_specs = self.default_memory_observable_specs(
            preset=preset,
            custom_specs=custom_specs,
        )
        return [self.parse_memory_observable(spec) for spec in deduped_specs]

    def fixed_point(self, tol: float = 1e-12, max_iter: int = 10000) -> np.ndarray:
        if self._fixed_point_cache is not None:
            return self._fixed_point_cache.copy()
        # Start from the maximally mixed state; for these small dense systems that
        # is a cheap and neutral initial guess.
        rho = np.eye(self.dim_memory, dtype=complex) / self.dim_memory
        for _ in range(max_iter):
            # Iterate the zero-input channel to find a stationary memory state.
            new_rho = self.channel(0.0, rho)
            new_rho = 0.5 * (new_rho + new_rho.conj().T)
            ensure_finite("fixed-point iterate", new_rho)
            tr = np.trace(new_rho)
            if abs(tr) > 1e-15:
                new_rho /= tr
            ensure_finite("normalized fixed-point iterate", new_rho)
            if np.linalg.norm(new_rho - rho, ord="fro") < tol:
                self._fixed_point_cache = new_rho.copy()
                return new_rho
            rho = new_rho
        self._fixed_point_cache = rho.copy()
        return rho


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
        self._initialize_common(params.n_memory, params.n_readout, params.reset_to_zero_state)
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
            H += p.jzz_memory * self._pair(self._memory_site(i), "Z", self._memory_site(j), "Z")
            if abs(p.jxx_memory) > 0:
                H += p.jxx_memory * self._pair(self._memory_site(i), "X", self._memory_site(j), "X")

        if abs(p.jzz_next_nearest) > 0 and p.n_memory >= 3:
            for i, j in self._memory_next_nearest_edges(p.periodic_memory_chain):
                H += p.jzz_next_nearest * self._pair(self._memory_site(i), "Z", self._memory_site(j), "Z")

        for i in range(p.n_memory):
            for a in range(p.n_readout):
                H += p.kz_memory_readout * self._pair(self._memory_site(i), "Z", self._readout_site(a), "Z")

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
        return checked_matmul("unitary eigendecomposition reconstruction", phased_evecs, evecs.conj().T)


@dataclass
class HaarRandomReservoirParameters:
    n_memory: int = 5
    n_readout: int = 1
    depth: int = 3
    seed: int = 1234
    input_qubit: int = 0
    encoding_qubits: int = 1
    ancilla_state: str = "zero"
    input_bias: float = 0.5
    input_scale: float = 0.5


class HaarRandomReservoirModel(ReservoirBase):
    """Random-circuit reservoir with GHZ-like input encoding.

    The circuit is fixed once at initialization. Each layer applies independent
    single-qubit Haar-random SU(2) rotations followed by a brickwork pattern of
    nearest-neighbor CNOT gates across the full memory+readout register.

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

    def __init__(self, params: HaarRandomReservoirParameters):
        if params.n_readout < 1:
            raise ValueError("HaarRandomReservoirModel requires n_readout >= 1")
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
        self._initialize_common(params.n_memory, params.n_readout, reset_to_zero_state=True)
        self._identity_total = pauli_string(self.n_total, tuple())
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

    def _haar_random_single_qubit(self, rng: np.random.Generator) -> np.ndarray:
        # Euler-angle construction for a Haar-random SU(2) gate.
        alpha = 2.0 * np.pi * rng.random()
        gamma = 2.0 * np.pi * rng.random()
        z = rng.random()
        beta = 2.0 * np.arccos(np.sqrt(z))
        return self._rotation_z(alpha) @ self._rotation_y(beta) @ self._rotation_z(gamma)

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
                gate = self._haar_random_single_qubit(rng)
                U = self._single_site_unitary(site, gate) @ U

            start = layer % 2
            # Alternate even and odd CNOT layers to produce a brickwork circuit
            # without trying to apply overlapping two-qubit gates at once.
            for control in range(start, self.n_total - 1, 2):
                target = control + 1
                U = self._cnot_gate(control, target) @ U

        return ensure_finite("haar-random circuit unitary", U)

    def _build_unitary(self, u: float) -> np.ndarray:
        # The circuit itself is fixed; only the injected readout state depends on u.
        return self._fixed_unitary

    def _zero_block_state(self, n_qubits: int) -> np.ndarray:
        state = np.zeros((2**n_qubits, 1), dtype=complex)
        state[0, 0] = 1.0
        return state

    def _ghz_like_state(self, p: float, n_qubits: int) -> np.ndarray:
        if n_qubits == 1:
            return np.array([[np.sqrt(p)], [np.sqrt(1.0 - p)]], dtype=complex)

        state = np.zeros((2**n_qubits, 1), dtype=complex)
        state[0, 0] = np.sqrt(p)
        state[-1, 0] = np.sqrt(1.0 - p)
        return state

    def _input_reset_state(self, u: float) -> np.ndarray:
        u = float(u)
        p = float(np.clip(self.params.input_bias + self.params.input_scale * u, 0.0, 1.0))
        left_qubits = self.params.input_qubit
        encoded_qubits = self.params.encoding_qubits
        right_qubits = self.n_readout - left_qubits - encoded_qubits

        left_state = self._zero_block_state(left_qubits) if left_qubits > 0 else None
        encoded_state = self._ghz_like_state(p, encoded_qubits)
        right_state = self._zero_block_state(right_qubits) if right_qubits > 0 else None

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

        U = self.unitary(u)
        U4 = U.reshape(self.dim_memory, self.dim_readout, self.dim_memory, self.dim_readout)
        reset_state = self._input_reset_state(u)
        evals, evecs = la.eigh(reset_state, check_finite=True)
        active = evals > 1e-15
        if not np.any(active):
            raise NumericalStabilityError("Input reset state has no positive eigenvalues")

        blocks = []
        for weight, psi in zip(evals[active], evecs[:, active].T, strict=False):
            contracted = np.einsum("arbi,i->arb", U4, psi, optimize=True)
            blocks.append(np.sqrt(weight) * np.transpose(contracted, (1, 0, 2)))
        kraus = ensure_finite(f"haar_kraus_operators(u={u})", np.concatenate(blocks, axis=0))
        self._cache_set(self._kraus_cache, u, kraus)
        return kraus


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
        self._initialize_common(params.n_memory, params.n_readout, params.reset_to_zero_state)
        self._rng = np.random.default_rng(params.seed)
        self._annihilation_ops = [self._build_annihilation(site) for site in range(self.n_total)]
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
        self._hamiltonian = ensure_finite("SYK Hamiltonian", self._syk4_h + self._syk2_h)
        self._hamiltonian = 0.5 * (self._hamiltonian + self._hamiltonian.conj().T)
        evals, evecs = la.eigh(self._hamiltonian, check_finite=True)
        self._hamiltonian_evals = ensure_finite("SYK Hamiltonian eigenvalues", evals)
        self._hamiltonian_evecs = ensure_finite("SYK Hamiltonian eigenvectors", evecs)
        phases = np.exp(-1j * self.params.tau * self._hamiltonian_evals)
        self._unitary = ensure_finite(
            "SYK unitary",
            (self._hamiltonian_evecs * phases[np.newaxis, :]) @ self._hamiltonian_evecs.conj().T,
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
        raw = _complex_normal_matrix(self._rng, (len(pair_indices), len(pair_indices)), scale=scale)
        coupling_matrix = 0.5 * (raw + raw.conj().T)

        pair_create = [self._creation_ops[i] @ self._creation_ops[j] for i, j in pair_indices]
        pair_annihilate = [self._annihilation_ops[k] @ self._annihilation_ops[l] for k, l in pair_indices]
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
        return float(np.clip(p, self.params.input_clip_eps, 1.0 - self.params.input_clip_eps))

    def input_state_vector(self, u: float) -> np.ndarray:
        p = self._encoded_probability(u)
        return np.array([np.sqrt(1.0 - p), np.sqrt(p)], dtype=complex)

    def _zero_block_state(self, n_qubits: int) -> np.ndarray:
        state = np.zeros((2**n_qubits, 1), dtype=complex)
        state[0, 0] = 1.0
        return state

    def _input_reset_state(self, u: float) -> np.ndarray:
        encoded_state = self.input_state_vector(u).reshape(2, 1)
        left_qubits = self.params.input_qubit
        right_qubits = self.n_readout - left_qubits - 1

        # Only one readout qubit carries the scalar input; any extra readout
        # qubits simply enlarge the environment and are reset to |0>.
        state = encoded_state
        if left_qubits > 0:
            state = np.kron(self._zero_block_state(left_qubits), state)
        if right_qubits > 0:
            state = np.kron(state, self._zero_block_state(right_qubits))
        return state @ state.conj().T

    def _build_unitary(self, u: float) -> np.ndarray:
        return self._unitary

    def kraus_operators(self, u: float) -> np.ndarray:
        u = float(u)
        cached = self._cache_get(self._kraus_cache, u)
        if cached is not None:
            return cached
        U = self.unitary(u)
        U4 = U.reshape(self.dim_memory, self.dim_readout, self.dim_memory, self.dim_readout)
        reset_state = self._input_reset_state(u)
        evals, evecs = la.eigh(reset_state, check_finite=True)
        active = evals > 1e-15
        if not np.any(active):
            raise NumericalStabilityError("SYK input reset state has no positive eigenvalues")

        blocks = []
        for weight, psi in zip(evals[active], evecs[:, active].T, strict=False):
            contracted = np.einsum("arbi,i->arb", U4, psi, optimize=True)
            blocks.append(np.sqrt(weight) * np.transpose(contracted, (1, 0, 2)))
        kraus = ensure_finite("SYK Kraus operators", np.concatenate(blocks, axis=0))
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
                    raise ValueError(f"Unsupported SYK number observable token '{token}'")
                site = int(token[1:])
                if not (0 <= site < self.n_memory):
                    raise ValueError(f"Observable token '{token}' is out of range for n_memory={self.n_memory}")
                # Number observables are multiplied directly, so strings like
                # N0*N1 become products of local occupation operators.
                out = out @ self._number_ops_memory[site]
            return out
        return super().parse_memory_observable(spec)

    def default_memory_observable_specs(
        self,
        preset: str = "occupation",
        custom_specs: Sequence[str] | None = None,
    ) -> List[str]:
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
            obs_specs = super().default_memory_observable_specs(preset=preset, custom_specs=None)
        if custom_specs:
            obs_specs.extend(custom_specs)
        return list(dict.fromkeys(obs_specs))

    def particle_number_sector_indices(self, total_particles: int) -> np.ndarray:
        if not (0 <= total_particles <= self.n_total):
            raise ValueError(f"total_particles={total_particles} must lie in [0, {self.n_total}]")
        return np.array([idx for idx in range(self.dim_total) if idx.bit_count() == total_particles], dtype=int)

    def sector_hamiltonian(self, total_particles: int) -> np.ndarray:
        indices = self.particle_number_sector_indices(total_particles)
        if indices.size == 0:
            raise NumericalStabilityError(f"No basis states in particle-number sector Np={total_particles}")
        return self._hamiltonian[np.ix_(indices, indices)]

    def mean_level_spacing_ratio(self, total_particles: int | None = None, central_fraction: float = 0.5) -> float:
        if total_particles is None:
            total_particles = self.n_total // 2
        sector_h = self.sector_hamiltonian(total_particles)
        evals = np.sort(la.eigvalsh(sector_h, check_finite=True))
        if evals.size < 3:
            raise NumericalStabilityError("Need at least 3 eigenvalues to compute spacing ratios.")
        fraction = float(np.clip(central_fraction, 0.0, 1.0))
        if 0.0 < fraction < 1.0:
            keep = max(3, int(round(fraction * evals.size)))
            start = max(0, (evals.size - keep) // 2)
            evals = evals[start : start + keep]
        spacings = np.diff(evals)
        ratios = np.minimum(spacings[:-1] / spacings[1:], spacings[1:] / spacings[:-1])
        ratios = ratios[np.isfinite(ratios)]
        if ratios.size == 0:
            raise NumericalStabilityError("No finite spacing ratios were produced.")
        return float(np.mean(ratios))
