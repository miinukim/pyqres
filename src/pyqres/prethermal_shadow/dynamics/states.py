"""Density-state composition and partial-trace helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .operators import kron_all


def density_zero(n_qubits: int) -> np.ndarray:
    rho = np.zeros((2 ** int(n_qubits), 2 ** int(n_qubits)), dtype=complex)
    rho[0, 0] = 1.0
    return rho


def density_plus(n_qubits: int) -> np.ndarray:
    one = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
    return kron_all([one for _ in range(int(n_qubits))])


def partial_trace_memory_first(
    rho: np.ndarray, n_memory: int, n_readout: int, keep: str
) -> np.ndarray:
    n_memory = int(n_memory)
    n_readout = int(n_readout)
    if keep == "memory":
        keep_qubits = tuple(range(n_memory))
    elif keep == "readout":
        keep_qubits = tuple(range(n_memory, n_memory + n_readout))
    else:
        raise ValueError("keep must be memory or readout.")
    return partial_trace_qubits(rho, n_memory + n_readout, keep_qubits)


def reorder_qubit_operator(
    operator: np.ndarray,
    current_order: Sequence[int],
    target_order: Sequence[int],
) -> np.ndarray:
    """Reorder matrix tensor factors from ``current_order`` to ``target_order``."""

    current = tuple(int(qubit) for qubit in current_order)
    target = tuple(int(qubit) for qubit in target_order)
    if (
        len(current) != len(target)
        or len(set(current)) != len(current)
        or len(set(target)) != len(target)
        or set(current) != set(target)
    ):
        raise ValueError(
            "current_order and target_order must contain the same unique qubits."
        )
    n_qubits = len(current)
    dim = 2**n_qubits
    matrix = np.asarray(operator, dtype=complex)
    if matrix.shape != (dim, dim):
        raise ValueError(f"operator must have shape {(dim, dim)}.")
    positions = {qubit: index for index, qubit in enumerate(current)}
    row_axes = [positions[qubit] for qubit in target]
    axes = [*row_axes, *(n_qubits + axis for axis in row_axes)]
    return matrix.reshape((2,) * (2 * n_qubits)).transpose(axes).reshape(dim, dim)


def combine_subsystem_states(
    rho_memory: np.ndarray,
    rho_readout: np.ndarray,
    memory_qubits: Sequence[int],
    readout_qubits: Sequence[int],
) -> np.ndarray:
    """Embed a memory/readout product state into physical qubit order."""

    memory = tuple(int(qubit) for qubit in memory_qubits)
    readout = tuple(int(qubit) for qubit in readout_qubits)
    subsystem_order = (*memory, *readout)
    return reorder_qubit_operator(
        np.kron(
            np.asarray(rho_memory, dtype=complex),
            np.asarray(rho_readout, dtype=complex),
        ),
        subsystem_order,
        tuple(range(len(subsystem_order))),
    )


def partial_trace_qubits(
    rho: np.ndarray,
    n_qubits: int,
    keep_qubits: Sequence[int],
) -> np.ndarray:
    """Trace out arbitrary qubits and order the result as ``keep_qubits``."""

    n_qubits = int(n_qubits)
    keep = tuple(int(qubit) for qubit in keep_qubits)
    if len(set(keep)) != len(keep):
        raise ValueError("keep_qubits must not contain duplicates.")
    if any(qubit < 0 or qubit >= n_qubits for qubit in keep):
        raise ValueError("keep_qubits contains an out-of-range qubit.")
    dim = 2**n_qubits
    matrix = np.asarray(rho, dtype=complex)
    if matrix.shape != (dim, dim):
        raise ValueError(f"rho must have shape {(dim, dim)}.")
    keep_set = set(keep)
    traced = tuple(qubit for qubit in range(n_qubits) if qubit not in keep_set)
    order = (*keep, *traced)
    tensor = reorder_qubit_operator(matrix, tuple(range(n_qubits)), order)
    dim_keep = 2 ** len(keep)
    dim_traced = 2 ** len(traced)
    tensor = tensor.reshape(dim_keep, dim_traced, dim_keep, dim_traced)
    return np.trace(tensor, axis1=1, axis2=3)


def project_density(rho: np.ndarray, *, tol: float = 1e-12) -> np.ndarray:
    rho = 0.5 * (
        np.asarray(rho, dtype=complex) + np.asarray(rho, dtype=complex).conj().T
    )
    tr = np.trace(rho)
    if abs(tr) > tol:
        rho = rho / tr
    return rho
