from __future__ import annotations

"""Dimension-analysis bridge for the prethermal shadow reservoir.

The physical reservoir keeps only the memory subsystem between steps. This
adapter exposes that induced memory channel to :mod:`pyqres.dim` and converts
the configured premeasurement features into equivalent memory observables at
one input expansion point.
"""

from collections import OrderedDict
from collections.abc import Sequence

import numpy as np
import scipy.linalg as la

from ..dim.linalg_utils import derivative_from_samples, ensure_finite
from ..dim.pauli import (
    default_pauli_observable_specs,
    pauli_basis_matrices,
    parse_pauli_observable_spec,
)
from .dynamics import pauli_string
from .reservoir import GlobalFloquetPartialShadowReservoir


class PrethermalShadowDimensionModel:
    """Expose prethermal STM dynamics through the standard dimension API.

    The channel is exactly the memory update used by
    :class:`GlobalFloquetPartialShadowReservoir`. For visibility analysis, a
    feature measured on the joint premeasurement state is pulled back to a
    memory observable at ``expansion_point``. Its expectation then agrees with
    the corresponding exact STM feature for every incoming memory state.

    The pulled-back observable is fixed at the expansion point. Consequently,
    the Volterra analysis measures how the persistent memory dynamics meets the
    local STM readout geometry; input derivatives of the readout map itself are
    intentionally outside this memory-visibility diagnostic.
    """

    def __init__(
        self,
        reservoir: GlobalFloquetPartialShadowReservoir,
        *,
        cache_max_entries: int | None = 32,
    ) -> None:
        self.reservoir = reservoir
        self.n_memory = int(reservoir.n_memory)
        self.n_readout = int(reservoir.n_readout)
        self.dim_memory = int(reservoir.dim_memory)
        self.dim_readout = int(reservoir.dim_readout)
        self._cache_max_entries = cache_max_entries
        self._kraus_block_cache: OrderedDict[float, np.ndarray] = OrderedDict()
        self._ptm_cache: OrderedDict[float, np.ndarray] = OrderedDict()
        self._memory_basis: tuple[np.ndarray, ...] | None = None
        self._fixed_point_cache: np.ndarray | None = None

    @property
    def memory_basis(self) -> tuple[np.ndarray, ...]:
        """Lazily materialize the full memory Pauli basis for dense analysis."""

        if self._memory_basis is None:
            self._memory_basis = pauli_basis_matrices(self.n_memory)
        return self._memory_basis

    def _cache_get(self, cache: OrderedDict[float, np.ndarray], key: float) -> np.ndarray | None:
        value = cache.get(key)
        if value is not None:
            cache.move_to_end(key)
        return value

    def _cache_set(self, cache: OrderedDict[float, np.ndarray], key: float, value: np.ndarray) -> None:
        cache[key] = value
        cache.move_to_end(key)
        if self._cache_max_entries is None:
            return
        if self._cache_max_entries <= 0:
            cache.clear()
            return
        while len(cache) > self._cache_max_entries:
            cache.popitem(last=False)

    def clear_caches(self) -> None:
        """Release cached channel samples and fixed-point data."""

        self._kraus_block_cache.clear()
        self._ptm_cache.clear()
        self._fixed_point_cache = None

    def unitary(self, u: float) -> np.ndarray:
        """Return the exact input-then-Floquet unitary used by one STM step."""

        return ensure_finite(
            "prethermal step unitary",
            self.reservoir.u_floquet_step @ self.reservoir._input_unitary(float(u)),
        )

    def _kraus_blocks(self, u: float) -> np.ndarray:
        """Return Kraus operators grouped by reset eigenstate and readout output."""

        key = float(u)
        cached = self._cache_get(self._kraus_block_cache, key)
        if cached is not None:
            return cached

        reset = self.reservoir._readout_reset_state()
        evals, evecs = la.eigh(reset, check_finite=True)
        active = evals > 1e-15
        if not np.any(active):
            raise ValueError("The readout reset state has no positive eigenvalues.")

        unitary = self.unitary(key).reshape(
            self.dim_memory,
            self.dim_readout,
            self.dim_memory,
            self.dim_readout,
        )
        blocks = []
        for weight, reset_vector in zip(evals[active], evecs[:, active].T, strict=False):
            contracted = np.einsum("arbi,i->rab", unitary, reset_vector, optimize=True)
            blocks.append(np.sqrt(weight) * contracted)
        out = ensure_finite("prethermal Kraus blocks", np.stack(blocks, axis=0))
        self._cache_set(self._kraus_block_cache, key, out)
        return out

    def kraus_operators(self, u: float) -> np.ndarray:
        """Return the induced memory-channel Kraus operators."""

        blocks = self._kraus_blocks(float(u))
        return blocks.reshape(-1, self.dim_memory, self.dim_memory)

    def channel(self, u: float, op_memory: np.ndarray) -> np.ndarray:
        """Apply the exact persistent-memory channel for one STM step."""

        blocks = self._kraus_blocks(float(u))
        operator = ensure_finite("memory operator", np.asarray(op_memory, dtype=complex))
        out = np.einsum("lrab,bc,lrdc->ad", blocks, operator, blocks.conj(), optimize=True)
        return ensure_finite("prethermal memory-channel output", out)

    def channel_adjoint(self, u: float, observable_memory: np.ndarray) -> np.ndarray:
        """Apply the Hilbert-Schmidt adjoint of the memory channel."""

        blocks = self._kraus_blocks(float(u))
        observable = ensure_finite(
            "memory observable",
            np.asarray(observable_memory, dtype=complex),
        )
        out = np.einsum("lrab,ac,lrcd->bd", blocks.conj(), observable, blocks, optimize=True)
        return ensure_finite("prethermal adjoint-channel output", out)

    def channel_derivative_adjoint(
        self,
        order: int,
        observable_memory: np.ndarray,
        u0: float = 0.0,
        fd_step: float = 5e-3,
        radius: int | None = None,
    ) -> np.ndarray:
        """Differentiate the adjoint memory channel around ``u0``."""

        if int(order) < 0:
            raise ValueError(f"Derivative order must be non-negative, got {order}.")
        if int(order) == 0:
            return self.channel_adjoint(float(u0), observable_memory)
        radius_value = int(radius) if radius is not None else max(2, int(order) + 1)
        points = list(range(-radius_value, radius_value + 1))
        samples = [
            self.channel_adjoint(float(u0) + point * float(fd_step), observable_memory)
            for point in points
        ]
        return ensure_finite(
            f"prethermal adjoint derivative order {order}",
            derivative_from_samples(samples, float(fd_step), int(order), points),
        )

    def _induced_readout_observable(self, readout_observable: np.ndarray, u: float) -> np.ndarray:
        blocks = self._kraus_blocks(float(u))
        out = np.einsum(
            "lrab,rs,lsac->bc",
            blocks.conj(),
            np.asarray(readout_observable, dtype=complex),
            blocks,
            optimize=True,
        )
        return ensure_finite("induced readout observable", 0.5 * (out + out.conj().T))

    def _induced_memory_observable(self, memory_observable: np.ndarray, u: float) -> np.ndarray:
        blocks = self._kraus_blocks(float(u))
        out = np.einsum(
            "lrab,ac,lrcd->bd",
            blocks.conj(),
            np.asarray(memory_observable, dtype=complex),
            blocks,
            optimize=True,
        )
        return ensure_finite("induced memory observable", 0.5 * (out + out.conj().T))

    def _induced_joint_observable(self, joint_observable: np.ndarray, u: float) -> np.ndarray:
        blocks = self._kraus_blocks(float(u))
        joint = np.asarray(joint_observable, dtype=complex).reshape(
            self.dim_memory,
            self.dim_readout,
            self.dim_memory,
            self.dim_readout,
        )
        out = np.einsum("lrab,arcs,lscd->bd", blocks.conj(), joint, blocks, optimize=True)
        return ensure_finite("induced joint observable", 0.5 * (out + out.conj().T))

    def feature_observables(self, expansion_point: float = 0.0) -> list[np.ndarray]:
        """Pull configured non-bias STM features back to memory observables."""

        scope = str(self.reservoir.feature_scope)
        observables = []
        for label in self.reservoir.pauli_labels:
            if scope == "readout":
                feature_op = pauli_string(self.n_readout, label)
                induced = self._induced_readout_observable(feature_op, float(expansion_point))
            elif scope == "memory":
                feature_op = pauli_string(self.n_memory, label)
                induced = self._induced_memory_observable(feature_op, float(expansion_point))
            else:
                feature_op = pauli_string(self.reservoir.n_qubits, label)
                induced = self._induced_joint_observable(feature_op, float(expansion_point))
            observables.append(induced)
        return observables

    def feature_observable_names(self) -> list[str]:
        """Return names aligned with :meth:`feature_observables`."""

        names = self.reservoir.get_feature_names()
        if self.reservoir.shadow_config.include_bias:
            return names[1:]
        return names

    def ptm(self, u: float) -> np.ndarray:
        """Build the full memory PTM for small-system dense validation."""

        key = float(u)
        cached = self._cache_get(self._ptm_cache, key)
        if cached is not None:
            return cached
        basis = self.memory_basis
        dim = self.dim_memory
        out = np.empty((len(basis), len(basis)), dtype=complex)
        for column, operator in enumerate(basis):
            evolved = self.channel(key, operator)
            out[:, column] = [np.trace(pauli.conj().T @ evolved) / dim for pauli in basis]
        out = ensure_finite(f"prethermal PTM(u={key})", out)
        self._cache_set(self._ptm_cache, key, out)
        return out

    def readout_matrix(self, observables: Sequence[np.ndarray]) -> np.ndarray:
        """Represent memory observables on the traceless Pauli basis."""

        basis = self.memory_basis[1:]
        dim = self.dim_memory
        return ensure_finite(
            "prethermal readout matrix",
            np.asarray(
                [[np.trace(observable @ pauli) / dim for pauli in basis] for observable in observables],
                dtype=complex,
            ),
        )

    def parse_memory_observable(self, spec: str) -> np.ndarray:
        return parse_pauli_observable_spec(self.n_memory, spec)

    def default_memory_observable_specs(
        self,
        preset: str = "z",
        custom_specs: Sequence[str] | None = None,
    ) -> list[str]:
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
        return [
            self.parse_memory_observable(spec)
            for spec in self.default_memory_observable_specs(preset, custom_specs)
        ]

    def fixed_point(self, tol: float = 1e-12, max_iter: int = 10000) -> np.ndarray:
        """Return the zero-input fixed point of the persistent memory channel."""

        if self._fixed_point_cache is not None:
            return self._fixed_point_cache.copy()
        rho = np.eye(self.dim_memory, dtype=complex) / self.dim_memory
        for _ in range(int(max_iter)):
            new_rho = self.channel(0.0, rho)
            new_rho = 0.5 * (new_rho + new_rho.conj().T)
            trace = np.trace(new_rho)
            if abs(trace) > 1e-15:
                new_rho /= trace
            if np.linalg.norm(new_rho - rho, ord="fro") < float(tol):
                rho = new_rho
                break
            rho = new_rho
        self._fixed_point_cache = ensure_finite("prethermal fixed point", rho.copy())
        return rho


__all__ = ["PrethermalShadowDimensionModel"]
