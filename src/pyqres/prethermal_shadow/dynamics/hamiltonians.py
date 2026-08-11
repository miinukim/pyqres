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


def build_step_unitary(
    cfg: GlobalFloquetConfig, h0: np.ndarray, drive: np.ndarray
) -> np.ndarray:
    u_period = build_fast_period_unitary(h0, drive, fast_period(cfg))
    return np.linalg.matrix_power(u_period, int(cfg.n_cycles_per_step))
