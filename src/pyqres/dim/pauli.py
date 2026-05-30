from __future__ import annotations

"""Dense Pauli-basis helpers used throughout the package.

The rest of the code works in a finite-dimensional operator basis rather than in
state-vector coordinates. For that reason, this module provides small cached
utilities for constructing tensor-product Pauli operators and a few standard
reference states. The implementation is intentionally dense and explicit: the
target system sizes in this project are still small enough that storing these
operators is simpler than introducing sparse or symbolic machinery.
"""

from functools import lru_cache
from itertools import product
from typing import Iterable, List, Sequence, Tuple

import numpy as np


PAULI_1Q = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


@lru_cache(maxsize=None)
def kron_all(labels: Tuple[str, ...]) -> np.ndarray:
    # Build a dense many-qubit operator from one-qubit Paulis in left-to-right order.
    out = np.array([[1.0 + 0.0j]])
    for label in labels:
        out = np.kron(out, PAULI_1Q[label])
    return out


@lru_cache(maxsize=None)
def pauli_basis(n_qubits: int) -> Tuple[Tuple[str, ...], ...]:
    # Enumerate the full tensor-product Pauli basis as label tuples like ("X", "I", "Z").
    return tuple(product(["I", "X", "Y", "Z"], repeat=n_qubits))


@lru_cache(maxsize=None)
def pauli_basis_matrices(n_qubits: int) -> Tuple[np.ndarray, ...]:
    # Materialize the full dense basis once, since downstream PTM code reuses it heavily.
    return tuple(kron_all(labels) for labels in pauli_basis(n_qubits))


@lru_cache(maxsize=None)
def basis_labels_as_strings(n_qubits: int) -> Tuple[str, ...]:
    # This is mainly useful for human-readable reporting or debugging of basis indices.
    return tuple("".join(labels) for labels in pauli_basis(n_qubits))


@lru_cache(maxsize=None)
def pauli_string(n_qubits: int, site_labels: Tuple[Tuple[int, str], ...]) -> np.ndarray:
    # Start from identity everywhere and replace only the requested sites.
    labels = ["I"] * n_qubits
    for idx, pauli in site_labels:
        labels[idx] = pauli
    return kron_all(tuple(labels))


@lru_cache(maxsize=None)
def single_site_pauli(n_qubits: int, site: int, pauli: str) -> np.ndarray:
    return pauli_string(n_qubits, ((site, pauli),))


def two_site_pauli(n_qubits: int, i: int, pauli_i: str, j: int, pauli_j: str) -> np.ndarray:
    # Sort the support so equivalent requests share the same cache entry in pauli_string().
    return pauli_string(n_qubits, tuple(sorted(((i, pauli_i), (j, pauli_j)))))


@lru_cache(maxsize=None)
def computational_zero_state(n_qubits: int) -> np.ndarray:
    # Return |0...0> as a column vector so callers can immediately form projectors.
    ket = np.zeros((2**n_qubits, 1), dtype=complex)
    ket[0, 0] = 1.0
    return ket


@lru_cache(maxsize=None)
def computational_zero_density(n_qubits: int) -> np.ndarray:
    # The reservoir-channel construction resets the readout subsystem to this state by default.
    ket = computational_zero_state(n_qubits)
    return ket @ ket.conj().T


@lru_cache(maxsize=None)
def maximally_mixed(n_qubits: int) -> np.ndarray:
    dim = 2**n_qubits
    return np.eye(dim, dtype=complex) / dim


@lru_cache(maxsize=None)
def hs_normalization_factor(n_qubits: int) -> float:
    return float(2**n_qubits)


@lru_cache(maxsize=None)
def traceless_basis_indices(n_qubits: int) -> Tuple[int, ...]:
    # Index 0 is the all-identity Pauli string, so the traceless sector starts at 1.
    return tuple(range(1, 4**n_qubits))
