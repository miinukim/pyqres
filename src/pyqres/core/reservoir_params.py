"""Parameter-generation helpers for dense reservoir Hamiltonians."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReservoirParams:
    """Random parameter generation for channel-map reservoirs.

    The exact and channel-map reservoirs consume explicit vectors/matrices:
    transverse fields `hx0_vec`, input-modulated longitudinal fields `hz1_vec`,
    and pairwise ZZ couplings `J_mat`. This dataclass provides reproducible
    defaults so experiments can be described by a compact seed plus graph type.
    """

    n_system: int = 4
    n_ancilla: int = 2
    tau: float = 1.0
    seed: int = 17462
    hx0_base: float = 1.0
    hz1_base: float = 1.0
    hx0_std: float = 0.3
    hz1_std: float = 0.3
    hx0_scale: float = 1.0
    hz1_scale: float = 1.0
    J_scale: float = 1.0
    graph_kind: str = "full"

    def n_qubits(self) -> int:
        """Return the joint memory+ancilla register size."""

        return self.n_system + self.n_ancilla

    def generate(self) -> dict:
        """Generate all Hamiltonian parameters as NumPy arrays.

        `J_mat` stores only the upper-triangular couplings because downstream
        Hamiltonian construction iterates over `i < j`. Keeping the lower
        triangle zero also makes serialized configs easier to inspect.
        """

        n = self.n_qubits()
        rs = np.random.RandomState(seed=self.seed)

        hx0 = (self.hx0_base + self.hx0_std * rs.randn(n)) * self.hx0_scale
        hz1 = (self.hz1_base + self.hz1_std * rs.randn(n)) * self.hz1_scale

        J_graph = np.zeros((n, n), dtype=float)
        graph_kind = self.graph_kind.lower()
        if graph_kind == "full":
            # Fully connected graph over all system and ancilla qubits.
            for i in range(n):
                for j in range(i + 1, n):
                    J_graph[i, j] = 1.0
        elif graph_kind == "rank6":
            # Historical six-qubit sparse graph used by earlier QRC rank tests.
            if n != 6:
                raise ValueError("graph_kind='rank6' is only defined for 6 total qubits.")
            J_graph = np.array(
                [
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0],
                ],
                dtype=float,
            )
        else:
            raise ValueError(f"Unsupported graph_kind '{self.graph_kind}'.")

        J = self.J_scale * (rs.rand(n, n) * J_graph)
        return {
            "hx0_vec": hx0,
            "hz1_vec": hz1,
            "J_mat": J,
            "tau": float(self.tau),
            "n_system": self.n_system,
            "n_ancilla": self.n_ancilla,
            "seed": self.seed,
        }
