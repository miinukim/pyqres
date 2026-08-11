"""Reservoir Hamiltonian parameter generators and presets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .operators import PauliTerm, dense_hamiltonian_matrix
from .specifications import HamiltonianSpec


@dataclass
class ReservoirParams:
    """Hamiltonian parameter generation for simulation reservoirs.

    The public output is always a backend-neutral pair H0_hamiltonian and
    H1_hamiltonian. The built-in Ising-type presets are compact generators for
    one such pair: H0 contains fixed transverse X fields and ZZ couplings,
    while H1 contains input-modulated Z fields.

    For broader Hamiltonians, set hamiltonian_kind="matrix" and provide
    h0_matrix/h1_matrix, or set hamiltonian_kind="pauli_terms" and provide
    h0_terms/h1_terms.
    """

    n_system: int = 6
    n_ancilla: int = 4
    tau: float = 1.0
    seed: int = 17462
    hamiltonian_kind: str = "ising"
    fixed_x_field_base: float = 1.0
    input_z_field_base: float = 1.0
    fixed_x_field_std: float = 0.3
    input_z_field_std: float = 0.3
    fixed_x_field_scale: float = 1.0
    input_z_field_scale: float = 1.0
    zz_coupling_scale: float = 1.0
    zz_connectivity: str = "open_chain"
    h0_matrix: Any | None = None
    h1_matrix: Any | None = None
    h0_hamiltonian: HamiltonianSpec | None = None
    h1_hamiltonian: HamiltonianSpec | None = None
    h0_terms: Sequence[PauliTerm | tuple[Any, Any] | Mapping[str, Any]] = field(
        default_factory=tuple
    )
    h1_terms: Sequence[PauliTerm | tuple[Any, Any] | Mapping[str, Any]] = field(
        default_factory=tuple
    )

    @classmethod
    def ising_type(cls, **kwargs: Any) -> ReservoirParams:
        """Create the built-in Ising-type Hamiltonian preset."""

        kwargs.setdefault("hamiltonian_kind", "ising")
        return cls(**kwargs)

    @classmethod
    def nisqrc_ising(cls, **kwargs: Any) -> ReservoirParams:
        """Create the Ising encoding used by Hu et al.'s NISQRC model.

        The defaults reproduce the fully connected Hamiltonian instance used
        for the paper's numerical channel-equalization experiment: fixed random
        ZZ couplings, transverse X fields, and input-modulated longitudinal Z
        fields. Coefficients are drawn once by :meth:`generate` and remain fixed
        for the complete input stream.
        """

        kwargs.setdefault("hamiltonian_kind", "ising")
        kwargs.setdefault("fixed_x_field_base", 2.0)
        kwargs.setdefault("fixed_x_field_std", 2.0)
        kwargs.setdefault("input_z_field_base", 0.5)
        kwargs.setdefault("input_z_field_std", 0.5)
        kwargs.setdefault("zz_coupling_scale", 1.0)
        kwargs.setdefault("zz_connectivity", "fully_connected")
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
    ) -> ReservoirParams:
        """Create a matrix-like Hamiltonian specification.

        h0_matrix and h1_matrix may be NumPy arrays, SciPy sparse matrices,
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
    ) -> ReservoirParams:
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

    def _metadata(self, hamiltonian_kind: str) -> dict[str, Any]:
        return {
            "tau": float(self.tau),
            "n_system": self.n_system,
            "n_ancilla": self.n_ancilla,
            "seed": self.seed,
            "hamiltonian_kind": hamiltonian_kind,
        }

    def _generate_matrix_hamiltonian(self) -> dict:
        h0 = self.h0_hamiltonian or HamiltonianSpec.from_matrix_like(
            self.n_qubits(), self.h0_matrix
        )
        h1 = self.h1_hamiltonian or HamiltonianSpec.from_matrix_like(
            self.n_qubits(), self.h1_matrix
        )
        return {
            "H0_hamiltonian": h0,
            "H1_hamiltonian": h1,
            "H0_matrix": self.h0_matrix,
            "H1_matrix": self.h1_matrix,
            **self._metadata("matrix"),
        }

    def _generate_pauli_terms_hamiltonian(self) -> dict:
        h0 = self.h0_hamiltonian or HamiltonianSpec.from_pauli_terms(
            self.n_qubits(), self.h0_terms
        )
        h1 = self.h1_hamiltonian or HamiltonianSpec.from_pauli_terms(
            self.n_qubits(), self.h1_terms
        )
        return {
            "H0_hamiltonian": h0,
            "H1_hamiltonian": h1,
            "H0_matrix": None,
            "H1_matrix": None,
            **self._metadata("pauli_terms"),
        }

    def _ising_hamiltonian_specs(
        self,
        fixed_x_field: np.ndarray,
        input_z_field: np.ndarray,
        zz_coupling: np.ndarray,
    ) -> tuple[HamiltonianSpec, HamiltonianSpec]:
        """Represent the generated Ising preset as symbolic Pauli terms."""

        h0_terms: list[PauliTerm] = []
        h1_terms: list[PauliTerm] = []
        n = self.n_qubits()
        for idx in range(n):
            h0_terms.append(PauliTerm(complex(fixed_x_field[idx]), ((idx, "X"),)))
            h1_terms.append(PauliTerm(complex(input_z_field[idx]), ((idx, "Z"),)))
        for i in range(n):
            for j in range(i + 1, n):
                jij = float(zz_coupling[i, j])
                if jij != 0.0:
                    h0_terms.append(PauliTerm(complex(jij), ((i, "Z"), (j, "Z"))))
        return (
            HamiltonianSpec(kind="ising", n_qubits=n, terms=tuple(h0_terms)),
            HamiltonianSpec(kind="ising", n_qubits=n, terms=tuple(h1_terms)),
        )

    def generate(self) -> dict:
        """Generate backend-neutral Hamiltonian specs for the requested kind."""

        kind = self.hamiltonian_kind.lower()
        if kind in {"matrix", "dense", "custom_matrix"}:
            return self._generate_matrix_hamiltonian()
        if kind in {"pauli", "pauli_terms", "terms"}:
            return self._generate_pauli_terms_hamiltonian()
        if kind not in {"ising", "ising_type"}:
            raise ValueError(f"Unsupported hamiltonian_kind '{self.hamiltonian_kind}'.")

        n = self.n_qubits()
        rs = np.random.RandomState(seed=self.seed)

        fixed_x_field = (
            self.fixed_x_field_base + self.fixed_x_field_std * rs.randn(n)
        ) * self.fixed_x_field_scale
        input_z_field = (
            self.input_z_field_base + self.input_z_field_std * rs.randn(n)
        ) * self.input_z_field_scale

        coupling_graph = np.zeros((n, n), dtype=float)
        connectivity = str(self.zz_connectivity).lower()
        if connectivity in {"open_chain", "chain"}:
            for i in range(n - 1):
                coupling_graph[i, i + 1] = 1.0
        elif connectivity in {"fully_connected", "all_to_all"}:
            coupling_graph[np.triu_indices(n, k=1)] = 1.0
        else:
            raise ValueError(
                "zz_connectivity must be one of: open_chain, chain, "
                "fully_connected, all_to_all"
            )

        zz_coupling = self.zz_coupling_scale * (rs.rand(n, n) * coupling_graph)
        h0, h1 = self._ising_hamiltonian_specs(
            fixed_x_field, input_z_field, zz_coupling
        )
        return {
            "H0_hamiltonian": h0,
            "H1_hamiltonian": h1,
            "H0_matrix": None,
            "H1_matrix": None,
            **self._metadata("ising"),
        }
