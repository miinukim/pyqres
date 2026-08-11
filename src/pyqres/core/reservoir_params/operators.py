"""Pauli-term normalization and dense/Qiskit conversion helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

PAULI_1Q = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


@dataclass(frozen=True)
class PauliTerm:
    """One term in a dense Pauli Hamiltonian.

    operators is a sequence of (site, pauli) pairs. Sites are indexed in the
    joint system+ancilla register. Example:

    python
    PauliTerm(0.7, ((0, "Z"), (2, "Z")))  # 0.7 * Z0 Z2

    """

    coefficient: complex
    operators: tuple[tuple[int, str], ...] = ()


def _kron_all(ops: Sequence[np.ndarray]) -> np.ndarray:
    out = np.array([[1.0 + 0.0j]])
    for op in ops:
        out = np.kron(out, op)
    return out


def _pauli_label(n_qubits: int, term: PauliTerm) -> str:
    labels = ["I"] * int(n_qubits)
    for site, pauli in term.operators:
        if not (0 <= int(site) < int(n_qubits)):
            raise ValueError(
                f"Pauli term site {site} is out of range for n_qubits={n_qubits}."
            )
        pauli = str(pauli).upper()
        if pauli not in PAULI_1Q:
            raise ValueError(f"Unsupported Pauli label '{pauli}'.")
        labels[int(site)] = pauli
    return "".join(labels)


def normalize_pauli_term(
    term: PauliTerm | tuple[Any, Any] | Mapping[str, Any],
) -> PauliTerm:
    """Accept PauliTerm, tuple, or dict input and normalize to PauliTerm."""

    if isinstance(term, PauliTerm):
        return term
    if isinstance(term, Mapping):
        coeff = term.get("coefficient", term.get("coeff", 1.0))
        operators = term.get("operators", term.get("paulis", ()))
        return PauliTerm(
            complex(coeff),
            tuple((int(site), str(pauli).upper()) for site, pauli in operators),
        )
    coeff, operators = term
    return PauliTerm(
        complex(coeff),
        tuple((int(site), str(pauli).upper()) for site, pauli in operators),
    )


def pauli_term_matrix(
    n_qubits: int, term: PauliTerm | tuple[Any, Any] | Mapping[str, Any]
) -> np.ndarray:
    """Convert one Pauli term into a dense matrix on n_qubits."""

    normalized = normalize_pauli_term(term)
    labels = _pauli_label(n_qubits, normalized)
    return complex(normalized.coefficient) * _kron_all(
        [PAULI_1Q[label] for label in labels]
    )


def pauli_terms_matrix(
    n_qubits: int,
    terms: Sequence[PauliTerm | tuple[Any, Any] | Mapping[str, Any]],
) -> np.ndarray:
    """Convert a list of Pauli terms into one dense Hermitian candidate matrix."""

    dim = 2 ** int(n_qubits)
    out = np.zeros((dim, dim), dtype=complex)
    for term in terms:
        out += pauli_term_matrix(n_qubits, term)
    return out


def pauli_terms_to_labels(
    n_qubits: int,
    terms: Sequence[PauliTerm | tuple[Any, Any] | Mapping[str, Any]],
) -> tuple[tuple[str, complex], ...]:
    """Normalize Pauli terms into (label, coefficient) pairs.

    Qubit labels follow the same left-to-right convention used by the dense
    Kronecker construction in this module and by Qiskit's Pauli string display.
    """

    normalized_terms = []
    for term in terms:
        normalized = normalize_pauli_term(term)
        normalized_terms.append(
            (_pauli_label(n_qubits, normalized), complex(normalized.coefficient))
        )
    return tuple(normalized_terms)


def pauli_terms_to_sparse_pauli_op(
    n_qubits: int,
    terms: Sequence[PauliTerm | tuple[Any, Any] | Mapping[str, Any]],
) -> Any:
    """Convert Pauli terms to Qiskit's SparsePauliOp without importing Qiskit globally."""

    try:
        from qiskit.quantum_info import SparsePauliOp
    except Exception as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "qiskit is required to build SparsePauliOp Hamiltonians."
        ) from exc

    label_terms = pauli_terms_to_labels(n_qubits, terms)
    if not label_terms:
        label_terms = (("I" * int(n_qubits), 0.0),)
    return SparsePauliOp.from_list(list(label_terms))


def dense_hamiltonian_matrix(matrix_like: Any) -> np.ndarray:
    """Convert a common Hamiltonian-like object into a dense complex matrix.

    The core package should not require optional simulation backends, so this
    helper intentionally uses duck typing. It accepts regular arrays, SciPy
    sparse matrices via toarray, and Qiskit quantum-info objects such as
    Operator/SparsePauliOp via to_matrix or data.
    """

    if hasattr(matrix_like, "to_dense") and callable(matrix_like.to_dense):
        return np.asarray(matrix_like.to_dense(), dtype=complex)

    candidate = matrix_like

    # Qiskit quantum-info classes expose to_matrix; SparsePauliOp and Operator
    # both fit this branch. Some implementations return sparse matrices, so a
    # later normalization pass still checks for toarray.
    if hasattr(candidate, "to_matrix") and callable(candidate.to_matrix):
        candidate = candidate.to_matrix()

    # A few operator wrappers expose to_operator() instead of a direct dense
    # conversion. Convert once and then use either .data or .to_matrix().
    elif hasattr(candidate, "to_operator") and callable(candidate.to_operator):
        operator = candidate.to_operator()
        if hasattr(operator, "data"):
            candidate = operator.data
        elif hasattr(operator, "to_matrix") and callable(operator.to_matrix):
            candidate = operator.to_matrix()

    # SciPy sparse matrices and sparse arrays expose toarray. Prefer this over
    # np.asarray, which would otherwise create an object array around them.
    if hasattr(candidate, "toarray") and callable(candidate.toarray):
        candidate = candidate.toarray()
    elif hasattr(candidate, "todense") and callable(candidate.todense):
        candidate = candidate.todense()
    elif hasattr(candidate, "data") and not isinstance(candidate, np.ndarray):
        # Qiskit Operator.data lands here if the object did not have
        # to_matrix; avoid applying this to ndarray, whose .data is a buffer.
        data = candidate.data
        if isinstance(data, np.ndarray):
            candidate = data

    out = np.asarray(candidate, dtype=complex)
    if out.ndim != 2:
        raise ValueError(
            f"Hamiltonian input must be a matrix, got array with shape {out.shape}."
        )
    return out
