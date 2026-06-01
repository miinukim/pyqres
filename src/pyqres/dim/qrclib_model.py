"""Adapter from the exact dense QRC core into the dimension-analysis interface."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .linalg_utils import ensure_finite
from .model import ReservoirBase

try:
    from pyqres.simulation.exact_qrc import ExactQRCModel, ExactQRCModelConfig
except Exception as exc:  # pragma: no cover
    ExactQRCModel = None  # type: ignore
    ExactQRCModelConfig = None  # type: ignore
    _QRCLIB_IMPORT_ERROR = exc
else:  # pragma: no cover
    _QRCLIB_IMPORT_ERROR = None


class QRCLibExactReservoirModel(ReservoirBase):
    """Dimension-analysis wrapper around pyqres' shared exact dense QRC core."""

    def __init__(self, config: "ExactQRCModelConfig | None" = None, core: "ExactQRCModel | None" = None):
        if ExactQRCModel is None:
            raise ImportError("pyqres.simulation must be importable to use QRCLibExactReservoirModel.") from _QRCLIB_IMPORT_ERROR
        if core is None and config is None:
            raise ValueError("Provide either a pyqres ExactQRCModelConfig or an ExactQRCModel instance.")
        self.core = core if core is not None else ExactQRCModel(config)  # type: ignore[arg-type]
        if self.core.control.post_measurement_mode != "reset":
            # A reset channel maps memory operators to memory operators. Without
            # reset, the effective state lives on the joint system and the PTM
            # assumptions in pyqres.dim do not apply.
            raise ValueError("pyqres PTM analysis requires post_measurement_mode='reset'.")
        self.params = self.core.cfg
        self._initialize_common(self.core.nS, self.core.nA, reset_to_zero_state=True)

    def _build_unitary(self, u: float) -> np.ndarray:
        return self.core.unitary(float(u))

    def channel(self, u: float, op_memory: np.ndarray) -> np.ndarray:
        out = self.core.system_channel(float(u), np.asarray(op_memory, dtype=complex))
        return ensure_finite("shared pyqres system channel output", out)

    def ptm(self, u: float) -> np.ndarray:
        u = float(u)
        cached = self._cache_get(self._ptm_cache, u)
        if cached is not None:
            return cached
        # The exact core already exposes the reduced memory channel. PTM
        # construction here just applies that channel to every Pauli basis
        # operator and projects the outputs back onto the same basis.
        # The pyqres exact core already exposes the reduced system channel, so PTM
        # construction here is just "apply channel to each basis operator, then project".
        outputs = np.stack([self.channel(u, basis_op) for basis_op in self.memory_basis], axis=0)
        t = np.einsum("mab,nab->mn", self._memory_basis_stack.conj(), outputs, optimize=True) / self.dim_memory
        t = ensure_finite(f"shared pyqres PTM(u={u})", t)
        self._cache_set(self._ptm_cache, u, t)
        return t

    def fixed_point(self, tol: float = 1e-12, max_iter: int = 10000) -> np.ndarray:
        if self._fixed_point_cache is not None:
            return self._fixed_point_cache.copy()
        rho = np.eye(self.dim_memory, dtype=complex) / self.dim_memory
        for _ in range(max_iter):
            new_rho = self.channel(0.0, rho)
            new_rho = 0.5 * (new_rho + new_rho.conj().T)
            tr = np.trace(new_rho)
            if abs(tr) > 1e-15:
                new_rho /= tr
            if np.linalg.norm(new_rho - rho, ord="fro") < tol:
                self._fixed_point_cache = new_rho.copy()
                return new_rho
            rho = new_rho
        self._fixed_point_cache = rho.copy()
        return rho


__all__ = ["QRCLibExactReservoirModel", "ExactQRCModel", "ExactQRCModelConfig"]
