"""Backend-neutral Hamiltonian specifications."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .operators import (
    PauliTerm,
    dense_hamiltonian_matrix,
    normalize_pauli_term,
    pauli_terms_matrix,
    pauli_terms_to_labels,
    pauli_terms_to_sparse_pauli_op,
)


@dataclass(frozen=True)
class HamiltonianSpec:
    """Backend-neutral Hamiltonian component.

    The spec preserves symbolic Pauli data for Qiskit/Aer-style backends while
    still giving dense exact simulation a single to_dense() conversion point.
    """

    kind: str
    n_qubits: int
    data: Any = None
    terms: tuple[PauliTerm, ...] = ()

    @classmethod
    def from_matrix_like(
        cls, n_qubits: int, matrix_like: Any | None
    ) -> HamiltonianSpec:
        """Wrap an arbitrary matrix/operator-like object without densifying it."""

        return cls(kind="matrix", n_qubits=int(n_qubits), data=matrix_like)

    @classmethod
    def from_pauli_terms(
        cls,
        n_qubits: int,
        terms: Sequence[PauliTerm | tuple[Any, Any] | Mapping[str, Any]],
    ) -> HamiltonianSpec:
        """Wrap Pauli terms in normalized form for later dense or Qiskit conversion."""

        normalized = tuple(normalize_pauli_term(term) for term in terms)
        pauli_terms_to_labels(n_qubits, normalized)
        return cls(kind="pauli_terms", n_qubits=int(n_qubits), terms=normalized)

    def to_dense(self) -> np.ndarray:
        """Materialize this Hamiltonian as a dense complex matrix."""

        dim = 2 ** int(self.n_qubits)
        if self.data is None and not self.terms:
            return np.zeros((dim, dim), dtype=complex)
        if self.kind in {"pauli", "pauli_terms", "ising"}:
            return pauli_terms_matrix(self.n_qubits, self.terms)
        return dense_hamiltonian_matrix(self.data)

    def to_sparse_pauli_op(self) -> Any:
        """Materialize this Hamiltonian as Qiskit's SparsePauliOp when possible."""

        if self.data is None and not self.terms:
            return pauli_terms_to_sparse_pauli_op(self.n_qubits, ())
        if self.kind in {"pauli", "pauli_terms", "ising"}:
            return pauli_terms_to_sparse_pauli_op(self.n_qubits, self.terms)
        if hasattr(self.data, "to_sparse_pauli_op") and callable(
            self.data.to_sparse_pauli_op
        ):
            return self.data.to_sparse_pauli_op()
        try:
            from qiskit.quantum_info import Operator, SparsePauliOp
        except Exception as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "qiskit is required to build SparsePauliOp Hamiltonians."
            ) from exc
        if isinstance(self.data, SparsePauliOp):
            return self.data
        return SparsePauliOp.from_operator(
            Operator(dense_hamiltonian_matrix(self.data))
        )
