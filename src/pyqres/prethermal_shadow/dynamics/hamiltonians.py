"""Hamiltonian and unitary construction for prethermal dynamics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.linalg as la

from ..config import GlobalFloquetConfig
from .operators import PauliTerm, pauli_terms_matrix


def build_global_h0(
    cfg: GlobalFloquetConfig,
    *,
    h: np.ndarray,
    jz: np.ndarray,
    jxy: np.ndarray,
    break_coeffs: np.ndarray,
    edges: Sequence[tuple[int, int]],
) -> np.ndarray:
    return pauli_terms_matrix(
        int(cfg.n_qubits),
        global_h0_pauli_terms(
            cfg,
            h=h,
            jz=jz,
            jxy=jxy,
            break_coeffs=break_coeffs,
            edges=edges,
        ),
    )


def global_h0_pauli_terms(
    cfg: GlobalFloquetConfig,
    *,
    h: np.ndarray,
    jz: np.ndarray,
    jxy: np.ndarray,
    break_coeffs: np.ndarray,
    edges: Sequence[tuple[int, int]],
) -> tuple[PauliTerm, ...]:
    """Return the static prethermal Hamiltonian as symbolic Pauli terms."""

    terms: list[PauliTerm] = []
    for i, coeff in enumerate(h):
        if float(coeff) != 0.0:
            terms.append((float(coeff), ((i, "Z"),)))
    for e, (i, j) in enumerate(edges):
        if float(jz[e]) != 0.0:
            terms.append((float(jz[e]), ((i, "Z"), (j, "Z"))))
        if float(jxy[e]) != 0.0:
            terms.append((float(jxy[e]), ((i, "X"), (j, "X"))))
            terms.append((float(jxy[e]), ((i, "Y"), (j, "Y"))))
    axis = str(cfg.break_axis).upper()
    for i, coeff in enumerate(break_coeffs):
        if float(coeff) != 0.0:
            terms.append((float(coeff), ((i, axis),)))
    return tuple(terms)


def build_drive_hamiltonian(
    cfg: GlobalFloquetConfig, drive_coeffs: np.ndarray
) -> np.ndarray:
    return pauli_terms_matrix(
        int(cfg.n_qubits),
        drive_pauli_terms(cfg, drive_coeffs),
    )


def drive_pauli_terms(
    cfg: GlobalFloquetConfig, drive_coeffs: np.ndarray
) -> tuple[PauliTerm, ...]:
    """Return the square-drive Hamiltonian as symbolic Pauli terms."""

    axis = str(cfg.drive_axis).upper()
    return tuple(
        (float(coeff), ((i, axis),))
        for i, coeff in enumerate(drive_coeffs)
        if float(coeff) != 0.0
    )


def fast_period(cfg: GlobalFloquetConfig) -> float:
    return 2.0 * np.pi / float(cfg.omega)


def step_duration(cfg: GlobalFloquetConfig) -> float:
    return int(cfg.n_cycles_per_step) * fast_period(cfg)


def build_fast_period_unitary(
    h0: np.ndarray, drive: np.ndarray, period: float
) -> np.ndarray:
    half = 0.5 * float(period)
    u_plus = la.expm(-1j * half * (h0 + drive))
    u_minus = la.expm(-1j * half * (h0 - drive))
    return u_minus @ u_plus


def _validated_segments(hamiltonians, durations):
    matrices = tuple(np.asarray(h, dtype=complex) for h in hamiltonians)
    times = np.asarray(durations, dtype=float)
    if not matrices or times.shape != (len(matrices),):
        raise ValueError("Specify one duration per Hamiltonian segment")
    if not np.isfinite(times).all() or np.any(times <= 0):
        raise ValueError("Segment durations must be finite and positive")
    shape = matrices[0].shape
    if len(shape) != 2 or shape[0] != shape[1] or shape[0] == 0:
        raise ValueError("Hamiltonians must be nonempty square matrices")
    for h in matrices:
        if h.shape != shape or not np.isfinite(h).all():
            raise ValueError("Hamiltonians must be finite and have equal shapes")
        if not np.allclose(h, h.conj().T, rtol=1e-12, atol=1e-12):
            raise ValueError("Hamiltonians must be Hermitian")
    return matrices, times


def build_piecewise_period_unitary(
    hamiltonians: Sequence[np.ndarray], durations: Sequence[float]
) -> np.ndarray:
    """Time-ordered propagation for arbitrary piecewise-constant controls.

    Inputs are in chronological order; the earliest exponential acts first
    and is rightmost in the returned product. Each matrix is the complete
    segment Hamiltonian, including any always-on static Hamiltonian.
    """
    matrices, times = _validated_segments(hamiltonians, durations)
    unitary = np.eye(matrices[0].shape[0], dtype=complex)
    for h, duration in zip(matrices, times, strict=True):
        unitary = la.expm(-1j * duration * h) @ unitary
    return unitary


def piecewise_effective_hamiltonian_first_order(
    hamiltonians: Sequence[np.ndarray], durations: Sequence[float]
) -> np.ndarray:
    """Stroboscopic Magnus Hamiltonian through first order in the period.

    H_eff = sum(dt_j H_j)/T - i/(2T) sum_{j>k} dt_j dt_k [H_j,H_k].
    The initial drive phase is the start of the first listed segment.
    """
    matrices, times = _validated_segments(hamiltonians, durations)
    period = float(times.sum())
    result = sum(t * h for t, h in zip(times, matrices, strict=True)) / period
    for j, h in enumerate(matrices):
        for k in range(j):
            result = result - .5j * times[j] * times[k] / period * (
                h @ matrices[k] - matrices[k] @ h
            )
    return result


def build_step_unitary(
    cfg: GlobalFloquetConfig, h0: np.ndarray, drive: np.ndarray
) -> np.ndarray:
    u_period = build_fast_period_unitary(h0, drive, fast_period(cfg))
    return np.linalg.matrix_power(u_period, int(cfg.n_cycles_per_step))
