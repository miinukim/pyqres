from __future__ import annotations

"""Diagnostics for global-Floquet partial-shadow reservoirs."""

from itertools import combinations

import numpy as np


def shadow_noise_threshold(n_features: int, shots: int, pauli_k: int, weak_strength: float, delta: float = 0.05) -> float:
    """Approximate finite-shot shadow threshold tau_{k,s}(N, delta)."""

    if int(shots) <= 0:
        raise ValueError("shots must be positive.")
    s = float(weak_strength)
    if not (0.0 < s <= 1.0):
        raise ValueError("weak_strength must lie in (0, 1].")
    m = max(int(n_features), 2)
    return float(np.sqrt(((3.0 / (s**2)) ** int(pauli_k)) * np.log(m / float(delta)) / int(shots)))


def lagged_volterra_design(inputs: np.ndarray, lag_horizon: int, max_order: int = 2) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Build scalar first/second-order lag monomials for empirical OVD."""

    values = np.asarray(inputs, dtype=float).reshape(-1)
    lag_horizon = int(lag_horizon)
    if lag_horizon <= 0:
        raise ValueError("lag_horizon must be positive.")
    if max_order not in {1, 2}:
        raise ValueError("only max_order 1 or 2 is supported.")
    if values.size <= lag_horizon:
        raise ValueError("input sequence must be longer than lag_horizon.")
    labels: list[tuple[int, ...]] = [(lag,) for lag in range(lag_horizon)]
    if max_order >= 2:
        labels.extend((a, b) for a in range(lag_horizon) for b in range(a + 1))
    rows = []
    for t in range(lag_horizon, values.size):
        row = []
        for label in labels:
            term = 1.0
            for lag in label:
                term *= values[t - lag]
            row.append(term)
        rows.append(row)
    return np.asarray(rows, dtype=float), labels


def empirical_volterra_ovd(
    inputs: np.ndarray,
    features: np.ndarray,
    *,
    lag_horizon: int,
    max_order: int = 2,
    ridge: float = 0.0,
    tol: float = 1e-10,
    finite_shadow_threshold: float | None = None,
) -> dict[str, object]:
    """Estimate OVD from fitted first/second-order Volterra kernel vectors."""

    design, labels = lagged_volterra_design(inputs, lag_horizon, max_order=max_order)
    y = np.asarray(features, dtype=float)
    if y.ndim != 2:
        raise ValueError("features must be a 2D matrix.")
    y = y[int(lag_horizon) :]
    if y.shape[0] != design.shape[0]:
        raise ValueError("inputs/features length mismatch after lag trimming.")
    if ridge > 0.0:
        gram = design.T @ design + float(ridge) * np.eye(design.shape[1])
        kernels = np.linalg.solve(gram, design.T @ y)
    else:
        kernels = np.linalg.lstsq(design, y, rcond=None)[0]
    hmat = kernels.T
    singular_values = np.linalg.svd(hmat, compute_uv=False)
    ovd_rank = int(np.sum(singular_values > float(tol)))
    finite_ovd = None
    if finite_shadow_threshold is not None:
        finite_ovd = int(np.sum(singular_values > (2.0 * float(finite_shadow_threshold) + float(tol))))
    return {
        "monomials": labels,
        "kernel_matrix": hmat,
        "singular_values": singular_values,
        "ovd_rank": ovd_rank,
        "finite_shadow_ovd": finite_ovd,
    }


def spectrum_summary(eigenvalues: np.ndarray, delta_t: float) -> dict[str, np.ndarray]:
    """Return abs, phase, and decay-rate arrays for memory-channel eigenvalues."""

    vals = np.asarray(eigenvalues, dtype=complex)
    abs_vals = np.abs(vals)
    clipped = np.clip(abs_vals, 1e-300, None)
    return {
        "lambda_abs": abs_vals,
        "lambda_phase": np.angle(vals),
        "decay_rate": -np.log(clipped) / float(delta_t),
    }


def low_weight_feature_count(n_qubits: int, pauli_k: int) -> int:
    """Return count of non-identity Pauli strings up to weight pauli_k."""

    total = 0
    for weight in range(1, int(pauli_k) + 1):
        total += (3**weight) * len(tuple(combinations(range(int(n_qubits)), weight)))
    return int(total)


__all__ = [
    "empirical_volterra_ovd",
    "lagged_volterra_design",
    "low_weight_feature_count",
    "shadow_noise_threshold",
    "spectrum_summary",
]
