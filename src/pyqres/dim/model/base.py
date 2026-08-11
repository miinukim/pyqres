"""Shared memory-channel and Pauli-transfer-matrix machinery."""

from collections import OrderedDict
from collections.abc import Sequence

import numpy as np
import scipy.linalg as la

from ..linalg_utils import (
    NumericalStabilityError,
    derivative_from_samples,
    ensure_finite,
)
from ..pauli import (
    computational_zero_density,
    default_pauli_observable_specs,
    parse_pauli_observable_spec,
    pauli_basis_matrices,
    single_site_pauli,
    two_site_pauli,
)


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

    def _cache_get(
        self, cache: OrderedDict[float, np.ndarray], key: float
    ) -> np.ndarray | None:
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
        return cached

    def _cache_set(
        self, cache: OrderedDict[float, np.ndarray], key: float, value: np.ndarray
    ) -> None:
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

        logical_sites = (
            tuple(int(site) for site in input_sites)
            if input_sites is not None
            else (int(input_site),)
        )
        if not logical_sites:
            raise ValueError("At least one input site is required")

        physical_sites: list[int] = []
        for site in logical_sites:
            if input_on_memory:
                if not (0 <= site < self.n_memory):
                    raise ValueError(
                        f"Input memory site {site} is out of range for n_memory={self.n_memory}"
                    )
                physical_sites.append(self._memory_site(site))
            else:
                if not (0 <= site < self.n_readout):
                    raise ValueError(
                        f"Input readout site {site} is out of range for n_readout={self.n_readout}"
                    )
                physical_sites.append(self._readout_site(site))
        return tuple(physical_sites)

    def _input_strength_prefactor(
        self, strength: float, n_sites: int, normalization: str
    ) -> float:
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
        kraus = self._kraus_from_reset_state(
            u, self.reset_state, name="kraus_operators"
        )
        self._cache_set(self._kraus_cache, u, kraus)
        return kraus

    def _kraus_from_reset_state(
        self, u: float, reset_state: np.ndarray, *, name: str
    ) -> np.ndarray:
        """Contract the joint unitary with one readout reset state."""

        U = self.unitary(u)
        U4 = U.reshape(
            self.dim_memory, self.dim_readout, self.dim_memory, self.dim_readout
        )
        evals, evecs = la.eigh(reset_state, check_finite=True)
        active = evals > 1e-15
        if not np.any(active):
            raise NumericalStabilityError(
                f"{name} reset state has no positive eigenvalues"
            )

        blocks = []
        for weight, psi in zip(evals[active], evecs[:, active].T, strict=False):
            # If the readout reset state is rho_R = sum_j w_j |psi_j><psi_j|, then
            # the effective memory channel is obtained by contracting U against
            # each populated readout eigenvector |psi_j>. The resulting blocks are
            # stacked into a conventional Kraus list over the memory subsystem.
            contracted = np.einsum("arbi,i->arb", U4, psi, optimize=True)
            blocks.append(np.sqrt(weight) * np.transpose(contracted, (1, 0, 2)))
        return ensure_finite(f"{name}(u={u})", np.concatenate(blocks, axis=0))

    def channel(self, u: float, op_memory: np.ndarray) -> np.ndarray:
        ensure_finite("memory operator", op_memory)
        kraus = self.kraus_operators(u)
        # Apply the CPTP map Phi_u(X) = sum_k K_k X K_k^\dagger on the memory subsystem.
        out = np.einsum("kab,bc,kdc->ad", kraus, op_memory, kraus.conj(), optimize=True)
        return ensure_finite("channel output", out)

    def channel_adjoint(self, u: float, observable_memory: np.ndarray) -> np.ndarray:
        ensure_finite("memory observable", observable_memory)
        kraus = self.kraus_operators(u)
        out = np.einsum(
            "kba,bc,kcd->ad", kraus.conj(), observable_memory, kraus, optimize=True
        )
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
        T = (
            np.einsum(
                "mab,nab->mn",
                self._memory_basis_stack.conj(),
                outputs,
                optimize=True,
            )
            / self.dim_memory
        )
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
        return parse_pauli_observable_spec(self.n_memory, spec)

    def default_memory_observable_specs(
        self,
        preset: str = "z",
        custom_specs: Sequence[str] | None = None,
    ) -> list[str]:
        # These presets are convenience libraries for common readout choices used
        # in the experiments. The return value is still just a list of strings so
        # callers can inspect or augment it before materializing dense operators.
        return default_pauli_observable_specs(
            self.n_memory,
            preset=preset,
            custom_specs=tuple(custom_specs or ()),
        )

    def default_memory_observables(
        self,
        preset: str = "z",
        custom_specs: Sequence[str] | None = None,
    ) -> list[np.ndarray]:
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
