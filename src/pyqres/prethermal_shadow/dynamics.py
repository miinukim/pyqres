from __future__ import annotations

"""Dense memory-dynamics builders for prethermal shadow reservoirs."""

import numpy as np
import scipy.linalg as la

from .config import ResolvedPrethermalShadowConfig


_PAULI = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def kron_all(ops: list[np.ndarray]) -> np.ndarray:
    """Return the Kronecker product of all operators."""

    out = np.array([[1.0 + 0.0j]])
    for op in ops:
        out = np.kron(out, op)
    return out


def single_memory_pauli(n_memory: int, site: int, pauli: str) -> np.ndarray:
    """Dense one-site Pauli on the memory Hilbert space."""

    ops = [_PAULI["I"]] * int(n_memory)
    ops[int(site)] = _PAULI[str(pauli).upper()]
    return kron_all(ops)


def pair_memory_pauli(n_memory: int, site_a: int, pauli_a: str, site_b: int, pauli_b: str) -> np.ndarray:
    """Dense two-site Pauli product on the memory Hilbert space."""

    ops = [_PAULI["I"]] * int(n_memory)
    ops[int(site_a)] = _PAULI[str(pauli_a).upper()]
    ops[int(site_b)] = _PAULI[str(pauli_b).upper()]
    return kron_all(ops)


def build_memory_h0(cfg: ResolvedPrethermalShadowConfig) -> np.ndarray:
    """Return dense H0 = H_slow + H_mix + H_break on memory only."""

    n_memory = int(cfg.base.n_memory)
    dim = 2**n_memory
    h0 = np.zeros((dim, dim), dtype=complex)
    for i, coeff in enumerate(cfg.h):
        h0 += float(coeff) * single_memory_pauli(n_memory, i, "Z")
    for edge, (i, j) in enumerate(cfg.edges):
        h0 += float(cfg.jz[edge]) * pair_memory_pauli(n_memory, i, "Z", j, "Z")
        h0 += float(cfg.jxy[edge]) * pair_memory_pauli(n_memory, i, "X", j, "X")
        h0 += float(cfg.jxy[edge]) * pair_memory_pauli(n_memory, i, "Y", j, "Y")
    for i, coeff in enumerate(cfg.x_break):
        h0 += float(coeff) * single_memory_pauli(n_memory, i, "X")
    return 0.5 * (h0 + h0.conj().T)


def build_drive_hamiltonian(cfg: ResolvedPrethermalShadowConfig) -> np.ndarray:
    """Return dense H_drive on memory only."""

    n_memory = int(cfg.base.n_memory)
    dim = 2**n_memory
    out = np.zeros((dim, dim), dtype=complex)
    axis = str(cfg.base.floquet.fast_drive.drive_axis).upper()
    for i, coeff in enumerate(cfg.drive):
        out += float(coeff) * single_memory_pauli(n_memory, i, axis)
    return 0.5 * (out + out.conj().T)


def build_legacy_effective_static_memory_unitary(cfg: ResolvedPrethermalShadowConfig) -> np.ndarray:
    """Return the old repeated effective-static memory unitary."""

    h0 = build_memory_h0(cfg)
    period = la.expm(-1j * float(cfg.base.floquet.tau) * h0)
    return np.linalg.matrix_power(period, int(cfg.base.floquet.n_floquet))


def build_fast_drive_period_unitary(cfg: ResolvedPrethermalShadowConfig) -> np.ndarray:
    """Return one binary high-frequency drive period U_F_fast."""

    if cfg.fast_drive_period is None:
        raise ValueError("fast_drive_period is only defined in floquet.mode='fast_drive'.")
    h0 = build_memory_h0(cfg)
    hd = build_drive_hamiltonian(cfg)
    half_period = 0.5 * float(cfg.fast_drive_period)
    u_plus = la.expm(-1j * half_period * (h0 + hd))
    u_minus = la.expm(-1j * half_period * (h0 - hd))
    return u_minus @ u_plus


def build_memory_step_unitary(cfg: ResolvedPrethermalShadowConfig) -> np.ndarray:
    """Return the full memory unitary for one reservoir input interval."""

    if cfg.base.floquet.mode == "effective_static":
        return build_legacy_effective_static_memory_unitary(cfg)
    u_fast = build_fast_drive_period_unitary(cfg)
    return np.linalg.matrix_power(u_fast, int(cfg.base.floquet.fast_drive.n_cycles_per_input))


__all__ = [
    "build_drive_hamiltonian",
    "build_fast_drive_period_unitary",
    "build_legacy_effective_static_memory_unitary",
    "build_memory_h0",
    "build_memory_step_unitary",
    "kron_all",
    "pair_memory_pauli",
    "single_memory_pauli",
]
