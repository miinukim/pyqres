from __future__ import annotations

"""Small linear-algebra helpers shared across model and analysis code.

These routines are intentionally lightweight wrappers around NumPy/SciPy. Their
main job is not to provide new algorithms, but to make the numerically delicate
parts of the codebase easier to read and easier to debug when something goes
wrong. In particular, many functions add shape/finite-value checks so failures
are reported close to the actual source of instability.
"""

from math import factorial
from typing import Sequence

import numpy as np
import scipy.linalg as la
import warnings


class NumericalStabilityError(RuntimeError):
    """Raised when a computation produces NaNs/Infs or otherwise unstable output."""

    pass


def _array_summary(name: str, arr: np.ndarray) -> str:
    # Use a compact summary so higher-level error messages remain readable even for huge arrays.
    arr = np.asarray(arr)
    finite = np.isfinite(arr)
    finite_count = int(np.count_nonzero(finite))
    total = int(arr.size)
    if finite_count:
        finite_vals = np.abs(arr[finite])
        max_abs = float(np.max(finite_vals))
        min_abs = float(np.min(finite_vals))
    else:
        max_abs = float("nan")
        min_abs = float("nan")
    return (
        f"{name}: shape={arr.shape}, dtype={arr.dtype}, "
        f"finite={finite_count}/{total}, min_abs={min_abs:.3e}, max_abs={max_abs:.3e}"
    )


def ensure_finite(name: str, arr: np.ndarray) -> np.ndarray:
    # Centralize NaN/Inf checks so failures include a compact diagnostic summary.
    if not np.all(np.isfinite(arr)):
        raise NumericalStabilityError(f"{name} contains non-finite values; {_array_summary(name, arr)}")
    return arr


def checked_matmul(name: str, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    # Wrap dense matmul to catch runtime warnings and attach operand summaries to errors.
    ensure_finite("left operand", left)
    ensure_finite("right operand", right)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            out = left @ right
    except FloatingPointError as exc:
        raise NumericalStabilityError(
            f"Numerical instability during {name}; "
            f"{_array_summary('left', left)}; {_array_summary('right', right)}"
        ) from exc
    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    if runtime_warnings:
        warning = runtime_warnings[0]
        location = f"{warning.filename}:{warning.lineno}" if warning.filename else "<unknown>"
        raise NumericalStabilityError(
            f"RuntimeWarning during {name} at {location}: {warning.message}; "
            f"{_array_summary('left', left)}; {_array_summary('right', right)}"
        )
    return ensure_finite(name, out)


def ensure_hermiticity(arr: np.ndarray):
    # Ensure Hermiticity a given matrix
    return 0.5 * (arr + arr.conj().T)


def partial_trace_last_subsystem(op: np.ndarray, dim_memory: int, dim_readout: int) -> np.ndarray:
    # Interpret the operator as memory x readout x memory x readout and trace out readout.
    reshaped = op.reshape(dim_memory, dim_readout, dim_memory, dim_readout)
    out = np.trace(reshaped, axis1=1, axis2=3)
    return ensure_finite("partial trace", out)



def operator_to_ptm_coords(op: np.ndarray, basis: Sequence[np.ndarray], dim_subsystem: int) -> np.ndarray:
    ensure_finite("operator for PTM projection", op)
    coords = []
    for idx, P in enumerate(basis):
        # Coordinates are Hilbert-Schmidt projections in the unnormalized Pauli basis.
        product = checked_matmul(f"PTM projection basis[{idx}] @ operator", P.conj().T, op)
        coords.append(np.trace(product) / dim_subsystem)
    return ensure_finite("PTM coordinates", np.array(coords, dtype=complex))



def ptm_coords_to_operator(coords: np.ndarray, basis: Sequence[np.ndarray], dim_subsystem: int) -> np.ndarray:
    # The basis is unnormalized, so reconstruction is the plain linear combination of basis elements.
    out = np.zeros_like(basis[0], dtype=complex)
    for c, P in zip(coords, basis):
        out += c * P
    return out


def hs_inner_product(left: np.ndarray, right: np.ndarray) -> complex:
    ensure_finite("left operator", left)
    ensure_finite("right operator", right)
    return complex(np.vdot(left.reshape(-1), right.reshape(-1)))


def hs_norm(op: np.ndarray) -> float:
    return float(np.sqrt(max(0.0, np.real_if_close(hs_inner_product(op, op)))))


def orthogonalize_operator(
    candidate: np.ndarray,
    basis: Sequence[np.ndarray],
    tol: float = 1e-10,
) -> np.ndarray | None:
    out = np.array(candidate, dtype=complex, copy=True)
    for existing in basis:
        # The basis is assumed to be already normalized, so each subtraction is
        # a plain Hilbert-Schmidt projection coefficient.
        out -= hs_inner_product(existing, out) * existing
    norm = hs_norm(out)
    if norm <= tol:
        return None
    return ensure_finite("orthonormalized operator", out / norm)



def orthonormal_basis_from_columns(mat: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    if mat.size == 0:
        return np.zeros((mat.shape[0], 0), dtype=complex)
    # Pivoted QR gives a numerically stable basis for the column span without forming Gram matrices.
    q, r, piv = la.qr(mat, mode="economic", pivoting=True)
    if r.size == 0:
        return np.zeros((mat.shape[0], 0), dtype=complex)
    diag = np.abs(np.diag(r))
    rank = int(np.sum(diag > tol))
    return q[:, :rank]



def matrix_rank(mat: np.ndarray, tol: float = 1e-10) -> int:
    if mat.size == 0:
        return 0
    s = la.svdvals(mat)
    return int(np.sum(s > tol))



def null_space(mat: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    return la.null_space(mat, rcond=tol)


def finite_difference_weights(order: int, points: Sequence[int]) -> np.ndarray:
    # Solve the standard Vandermonde system for a finite-difference stencil centered at zero.
    # The resulting weights satisfy sum_j w_j f(p_j h) ~= h^order f^(order)(0).
    n = len(points)
    a = np.zeros((n, n), dtype=float)
    b = np.zeros(n, dtype=float)
    for row in range(n):
        a[row, :] = [p**row for p in points]
    b[order] = factorial(order)
    return np.linalg.solve(a, b)



def derivative_from_samples(samples: Sequence[np.ndarray], step: float, order: int, points: Sequence[int]) -> np.ndarray:
    # This helper treats each sample as an array-valued function value and applies
    # the same scalar finite-difference stencil to every entry.
    weights = finite_difference_weights(order, points)
    out = np.zeros_like(samples[0], dtype=complex)
    for w, sample in zip(weights, samples):
        # Apply the scalar stencil weights entrywise to matrix-valued samples.
        out += w * sample
    out /= step**order
    return out



def principal_angles(subspace_a: np.ndarray, subspace_b: np.ndarray) -> np.ndarray:
    if subspace_a.size == 0 or subspace_b.size == 0:
        return np.array([], dtype=float)
    qa = orthonormal_basis_from_columns(subspace_a)
    qb = orthonormal_basis_from_columns(subspace_b)
    if qa.shape[1] == 0 or qb.shape[1] == 0:
        return np.array([], dtype=float)
    # Singular values of Qa^* Qb are cosines of the principal angles.
    s = la.svdvals(qa.conj().T @ qb)
    s = np.clip(np.real_if_close(s), -1.0, 1.0)
    return np.arccos(s)
