"""Parameter-generation helpers for reservoir Hamiltonians.

`ReservoirParams` supports three levels of Hamiltonian specification:

1. the built-in Ising-type preset used by the original QRC experiments
2. explicit matrix-like `H0`/`H1` objects, including NumPy arrays, SciPy
   sparse matrices, and Qiskit quantum-info operators
3. explicit Pauli-term lists, which can stay symbolic until a backend chooses a
   concrete representation

The simulation core evolves Hamiltonian inputs as `H(u) = H0 + input_scale*u*H1`.
For historical configs, the Ising preset still also returns `hx0_vec`,
`hz1_vec`, and `J_mat`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

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

    `operators` is a sequence of `(site, pauli)` pairs. Sites are indexed in the
    joint system+ancilla register. Example:

    ```python
    PauliTerm(0.7, ((0, "Z"), (2, "Z")))  # 0.7 * Z0 Z2
    ```
    """

    coefficient: complex
    operators: tuple[tuple[int, str], ...] = ()


def _kron_all(ops: Sequence[np.ndarray]) -> np.ndarray:
    out = np.array([[1.0 + 0.0j]])
    for op in ops:
        out = np.kron(out, op)
    return out


def normalize_pauli_term(term: PauliTerm | tuple[Any, Any] | Mapping[str, Any]) -> PauliTerm:
    """Accept PauliTerm, tuple, or dict input and normalize to `PauliTerm`."""

    if isinstance(term, PauliTerm):
        return term
    if isinstance(term, Mapping):
        coeff = term.get("coefficient", term.get("coeff", 1.0))
        operators = term.get("operators", term.get("paulis", ()))
        return PauliTerm(complex(coeff), tuple((int(site), str(pauli).upper()) for site, pauli in operators))
    coeff, operators = term
    return PauliTerm(complex(coeff), tuple((int(site), str(pauli).upper()) for site, pauli in operators))


def pauli_term_matrix(n_qubits: int, term: PauliTerm | tuple[Any, Any] | Mapping[str, Any]) -> np.ndarray:
    """Convert one Pauli term into a dense matrix on `n_qubits`."""

    normalized = normalize_pauli_term(term)
    labels = ["I"] * int(n_qubits)
    for site, pauli in normalized.operators:
        if not (0 <= int(site) < int(n_qubits)):
            raise ValueError(f"Pauli term site {site} is out of range for n_qubits={n_qubits}.")
        pauli = str(pauli).upper()
        if pauli not in PAULI_1Q:
            raise ValueError(f"Unsupported Pauli label '{pauli}'.")
        labels[int(site)] = pauli
    return complex(normalized.coefficient) * _kron_all([PAULI_1Q[label] for label in labels])


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
    """Normalize Pauli terms into `(label, coefficient)` pairs.

    Qubit labels follow the same left-to-right convention used by the dense
    Kronecker construction in this module and by Qiskit's Pauli string display.
    """

    normalized_terms = []
    for term in terms:
        normalized = normalize_pauli_term(term)
        labels = ["I"] * int(n_qubits)
        for site, pauli in normalized.operators:
            if not (0 <= int(site) < int(n_qubits)):
                raise ValueError(f"Pauli term site {site} is out of range for n_qubits={n_qubits}.")
            pauli = str(pauli).upper()
            if pauli not in PAULI_1Q:
                raise ValueError(f"Unsupported Pauli label '{pauli}'.")
            labels[int(site)] = pauli
        normalized_terms.append(("".join(labels), complex(normalized.coefficient)))
    return tuple(normalized_terms)


def pauli_terms_to_sparse_pauli_op(
    n_qubits: int,
    terms: Sequence[PauliTerm | tuple[Any, Any] | Mapping[str, Any]],
) -> Any:
    """Convert Pauli terms to Qiskit's `SparsePauliOp` without importing Qiskit globally."""

    try:
        from qiskit.quantum_info import SparsePauliOp
    except Exception as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("qiskit is required to build SparsePauliOp Hamiltonians.") from exc

    label_terms = pauli_terms_to_labels(n_qubits, terms)
    if not label_terms:
        label_terms = (("I" * int(n_qubits), 0.0),)
    return SparsePauliOp.from_list(list(label_terms))


def dense_hamiltonian_matrix(matrix_like: Any) -> np.ndarray:
    """Convert a common Hamiltonian-like object into a dense complex matrix.

    The core package should not require optional simulation backends, so this
    helper intentionally uses duck typing. It accepts regular arrays, SciPy
    sparse matrices via `toarray`, and Qiskit quantum-info objects such as
    `Operator`/`SparsePauliOp` via `to_matrix` or `data`.
    """

    if hasattr(matrix_like, "to_dense") and callable(matrix_like.to_dense):
        return np.asarray(matrix_like.to_dense(), dtype=complex)

    candidate = matrix_like

    # Qiskit quantum-info classes expose `to_matrix`; SparsePauliOp and Operator
    # both fit this branch. Some implementations return sparse matrices, so a
    # later normalization pass still checks for `toarray`.
    if hasattr(candidate, "to_matrix") and callable(candidate.to_matrix):
        candidate = candidate.to_matrix()

    # A few operator wrappers expose `to_operator()` instead of a direct dense
    # conversion. Convert once and then use either `.data` or `.to_matrix()`.
    elif hasattr(candidate, "to_operator") and callable(candidate.to_operator):
        operator = candidate.to_operator()
        if hasattr(operator, "data"):
            candidate = operator.data
        elif hasattr(operator, "to_matrix") and callable(operator.to_matrix):
            candidate = operator.to_matrix()

    # SciPy sparse matrices and sparse arrays expose `toarray`. Prefer this over
    # `np.asarray`, which would otherwise create an object array around them.
    if hasattr(candidate, "toarray") and callable(candidate.toarray):
        candidate = candidate.toarray()
    elif hasattr(candidate, "todense") and callable(candidate.todense):
        candidate = candidate.todense()
    elif hasattr(candidate, "data") and not isinstance(candidate, np.ndarray):
        # Qiskit Operator.data lands here if the object did not have
        # `to_matrix`; avoid applying this to ndarray, whose `.data` is a buffer.
        data = candidate.data
        if isinstance(data, np.ndarray):
            candidate = data

    out = np.asarray(candidate, dtype=complex)
    if out.ndim != 2:
        raise ValueError(f"Hamiltonian input must be a matrix, got array with shape {out.shape}.")
    return out


@dataclass(frozen=True)
class HamiltonianSpec:
    """Backend-neutral Hamiltonian component.

    The spec preserves symbolic Pauli data for Qiskit/Aer-style backends while
    still giving dense exact simulation a single `to_dense()` conversion point.
    """

    kind: str
    n_qubits: int
    data: Any = None
    terms: tuple[PauliTerm, ...] = ()

    @classmethod
    def from_matrix_like(cls, n_qubits: int, matrix_like: Any | None) -> "HamiltonianSpec":
        """Wrap an arbitrary matrix/operator-like object without densifying it."""

        return cls(kind="matrix", n_qubits=int(n_qubits), data=matrix_like)

    @classmethod
    def from_pauli_terms(
        cls,
        n_qubits: int,
        terms: Sequence[PauliTerm | tuple[Any, Any] | Mapping[str, Any]],
    ) -> "HamiltonianSpec":
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
        """Materialize this Hamiltonian as Qiskit's `SparsePauliOp` when possible."""

        if self.data is None and not self.terms:
            return pauli_terms_to_sparse_pauli_op(self.n_qubits, ())
        if self.kind in {"pauli", "pauli_terms", "ising"}:
            return pauli_terms_to_sparse_pauli_op(self.n_qubits, self.terms)
        if hasattr(self.data, "to_sparse_pauli_op") and callable(self.data.to_sparse_pauli_op):
            return self.data.to_sparse_pauli_op()
        try:
            from qiskit.quantum_info import Operator, SparsePauliOp
        except Exception as exc:  # pragma: no cover - depends on optional extra
            raise ImportError("qiskit is required to build SparsePauliOp Hamiltonians.") from exc
        if isinstance(self.data, SparsePauliOp):
            return self.data
        return SparsePauliOp.from_operator(Operator(dense_hamiltonian_matrix(self.data)))


@dataclass
class ReservoirParams:
    """Hamiltonian parameter generation for simulation reservoirs.

    The Ising-type preset produces explicit vectors/matrices:
    transverse fields `hx0_vec`, input-modulated longitudinal fields `hz1_vec`,
    and nearest-neighbor open-boundary ZZ couplings `J_mat`. This dataclass
    provides reproducible defaults so experiments can be described compactly.

    For broader Hamiltonians, set `hamiltonian_kind="matrix"` and provide
    `h0_matrix`/`h1_matrix`, or set `hamiltonian_kind="pauli_terms"` and provide
    `h0_terms`/`h1_terms`.
    """

    n_system: int = 6
    n_ancilla: int = 4
    tau: float = 1.0
    seed: int = 17462
    hamiltonian_kind: str = "ising"
    hx0_base: float = 1.0
    hz1_base: float = 1.0
    hx0_std: float = 0.3
    hz1_std: float = 0.3
    hx0_scale: float = 1.0
    hz1_scale: float = 1.0
    J_scale: float = 1.0
    h0_matrix: Any | None = None
    h1_matrix: Any | None = None
    h0_hamiltonian: HamiltonianSpec | None = None
    h1_hamiltonian: HamiltonianSpec | None = None
    h0_terms: Sequence[PauliTerm | tuple[Any, Any] | Mapping[str, Any]] = field(default_factory=tuple)
    h1_terms: Sequence[PauliTerm | tuple[Any, Any] | Mapping[str, Any]] = field(default_factory=tuple)

    @classmethod
    def ising_type(cls, **kwargs: Any) -> "ReservoirParams":
        """Create the built-in Ising-type Hamiltonian preset."""

        kwargs.setdefault("hamiltonian_kind", "ising")
        return cls(**kwargs)

    @classmethod
    def from_matrices(
        cls,
        *,
        n_system: int,
        n_ancilla: int,
        h0_matrix: Any,
        h1_matrix: Any | None = None,
        tau: float = 1.0,
        seed: int = 17462,
    ) -> "ReservoirParams":
        """Create a matrix-like Hamiltonian specification.

        `h0_matrix` and `h1_matrix` may be NumPy arrays, SciPy sparse matrices,
        or Qiskit quantum-info operators. They are wrapped without densifying so
        downstream backends can choose dense, sparse, or Qiskit-native execution.
        """

        return cls(
            n_system=n_system,
            n_ancilla=n_ancilla,
            tau=tau,
            seed=seed,
            hamiltonian_kind="matrix",
            h0_matrix=h0_matrix,
            h1_matrix=h1_matrix,
        )

    @classmethod
    def from_pauli_terms(
        cls,
        *,
        n_system: int,
        n_ancilla: int,
        h0_terms: Sequence[PauliTerm | tuple[Any, Any] | Mapping[str, Any]],
        h1_terms: Sequence[PauliTerm | tuple[Any, Any] | Mapping[str, Any]] = (),
        tau: float = 1.0,
        seed: int = 17462,
    ) -> "ReservoirParams":
        """Create a Hamiltonian specification from Pauli-term lists."""

        return cls(
            n_system=n_system,
            n_ancilla=n_ancilla,
            tau=tau,
            seed=seed,
            hamiltonian_kind="pauli_terms",
            h0_terms=tuple(h0_terms),
            h1_terms=tuple(h1_terms),
        )

    def n_qubits(self) -> int:
        """Return the joint memory+ancilla register size."""

        return self.n_system + self.n_ancilla

    def _validate_dense_matrix(self, name: str, matrix: Any | None) -> np.ndarray:
        dim = 2 ** self.n_qubits()
        if matrix is None:
            return np.zeros((dim, dim), dtype=complex)
        out = dense_hamiltonian_matrix(matrix)
        if out.shape != (dim, dim):
            raise ValueError(f"{name} must have shape {(dim, dim)}, got {out.shape}.")
        if not np.allclose(out, out.conj().T, atol=1e-10):
            raise ValueError(f"{name} must be Hermitian.")
        return out

    def _generate_matrix_hamiltonian(self) -> dict:
        h0 = self.h0_hamiltonian or HamiltonianSpec.from_matrix_like(self.n_qubits(), self.h0_matrix)
        h1 = self.h1_hamiltonian or HamiltonianSpec.from_matrix_like(self.n_qubits(), self.h1_matrix)
        return {
            "H0_hamiltonian": h0,
            "H1_hamiltonian": h1,
            "H0_matrix": self.h0_matrix,
            "H1_matrix": self.h1_matrix,
            "tau": float(self.tau),
            "n_system": self.n_system,
            "n_ancilla": self.n_ancilla,
            "seed": self.seed,
            "hamiltonian_kind": "matrix",
        }

    def _generate_pauli_terms_hamiltonian(self) -> dict:
        h0 = self.h0_hamiltonian or HamiltonianSpec.from_pauli_terms(self.n_qubits(), self.h0_terms)
        h1 = self.h1_hamiltonian or HamiltonianSpec.from_pauli_terms(self.n_qubits(), self.h1_terms)
        return {
            "H0_hamiltonian": h0,
            "H1_hamiltonian": h1,
            "H0_matrix": None,
            "H1_matrix": None,
            "tau": float(self.tau),
            "n_system": self.n_system,
            "n_ancilla": self.n_ancilla,
            "seed": self.seed,
            "hamiltonian_kind": "pauli_terms",
        }

    def _ising_hamiltonian_specs(self, hx0: np.ndarray, hz1: np.ndarray, J: np.ndarray) -> tuple[HamiltonianSpec, HamiltonianSpec]:
        """Represent the generated Ising preset as symbolic Pauli terms."""

        h0_terms: list[PauliTerm] = []
        h1_terms: list[PauliTerm] = []
        n = self.n_qubits()
        for idx in range(n):
            h0_terms.append(PauliTerm(complex(hx0[idx]), ((idx, "X"),)))
            h1_terms.append(PauliTerm(complex(hz1[idx]), ((idx, "Z"),)))
        for i in range(n):
            for j in range(i + 1, n):
                jij = float(J[i, j])
                if jij != 0.0:
                    h0_terms.append(PauliTerm(complex(jij), ((i, "Z"), (j, "Z"))))
        return (
            HamiltonianSpec(kind="ising", n_qubits=n, terms=tuple(h0_terms)),
            HamiltonianSpec(kind="ising", n_qubits=n, terms=tuple(h1_terms)),
        )

    def generate(self) -> dict:
        """Generate Hamiltonian parameters and backend-neutral Hamiltonian specs.

        `J_mat` stores only the upper-triangular couplings because downstream
        Hamiltonian construction iterates over `i < j`. Keeping the lower
        triangle zero also makes serialized configs easier to inspect.
        """

        kind = self.hamiltonian_kind.lower()
        if kind in {"matrix", "dense", "custom_matrix"}:
            return self._generate_matrix_hamiltonian()
        if kind in {"pauli", "pauli_terms", "terms"}:
            return self._generate_pauli_terms_hamiltonian()
        if kind not in {"ising", "ising_type", "ising-like", "ising_like"}:
            raise ValueError(f"Unsupported hamiltonian_kind '{self.hamiltonian_kind}'.")

        n = self.n_qubits()
        rs = np.random.RandomState(seed=self.seed)

        hx0 = (self.hx0_base + self.hx0_std * rs.randn(n)) * self.hx0_scale
        hz1 = (self.hz1_base + self.hz1_std * rs.randn(n)) * self.hz1_scale

        # Standard open-boundary Ising chain: only nearest-neighbor ZZ couplings
        # are active, with no wrap-around edge between the last and first qubit.
        J_graph = np.zeros((n, n), dtype=float)
        for i in range(n - 1):
            J_graph[i, i + 1] = 1.0

        J = self.J_scale * (rs.rand(n, n) * J_graph)
        h0, h1 = self._ising_hamiltonian_specs(hx0, hz1, J)
        return {
            "hx0_vec": hx0,
            "hz1_vec": hz1,
            "J_mat": J,
            "H0_hamiltonian": h0,
            "H1_hamiltonian": h1,
            "H0_matrix": None,
            "H1_matrix": None,
            "tau": float(self.tau),
            "n_system": self.n_system,
            "n_ancilla": self.n_ancilla,
            "seed": self.seed,
            "hamiltonian_kind": "ising",
        }
