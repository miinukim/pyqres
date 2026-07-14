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
from itertools import combinations, product
from typing import Tuple

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


def parse_pauli_observable_spec(n_qubits: int, spec: str) -> np.ndarray:
    """Parse specs such as ``Z0`` or ``X0*Z2`` into dense memory operators."""

    cleaned = str(spec).replace(" ", "")
    if not cleaned:
        raise ValueError("Observable spec must be non-empty")

    factors = []
    for token in cleaned.split("*"):
        pauli = token[0].upper()
        if pauli not in {"X", "Y", "Z"}:
            raise ValueError(f"Unsupported Pauli observable token '{token}'")
        try:
            site = int(token[1:])
        except ValueError as exc:
            raise ValueError(f"Observable token '{token}' must have an integer site index") from exc
        if not (0 <= site < n_qubits):
            raise ValueError(f"Observable token '{token}' is out of range for n_qubits={n_qubits}")
        factors.append((site, pauli))
    return pauli_string(n_qubits, tuple(sorted(factors)))


def single_site_observable_specs(n_qubits: int, paulis: Tuple[str, ...]) -> list[str]:
    return [f"{pauli}{site}" for pauli in paulis for site in range(n_qubits)]


def pair_observable_specs(n_qubits: int, paulis_left: Tuple[str, ...], paulis_right: Tuple[str, ...]) -> list[str]:
    return [
        f"{left_pauli}{left_site}*{right_pauli}{right_site}"
        for left_site, right_site in combinations(range(n_qubits), 2)
        for left_pauli, right_pauli in product(paulis_left, paulis_right)
    ]


def nearest_neighbor_observable_specs(n_qubits: int, paulis_left: Tuple[str, ...], paulis_right: Tuple[str, ...]) -> list[str]:
    return [
        f"{left_pauli}{left_site}*{right_pauli}{left_site + 1}"
        for left_site in range(n_qubits - 1)
        for left_pauli, right_pauli in product(paulis_left, paulis_right)
    ]


def default_pauli_observable_specs(
    n_qubits: int,
    preset: str = "z",
    custom_specs: Tuple[str, ...] = (),
    *,
    include_extended_pairs: bool = True,
) -> list[str]:
    """Return named Pauli observable presets shared by dim models and streams."""

    preset_key = preset.lower()
    if preset_key in {"x", "y", "z"}:
        obs_specs = single_site_observable_specs(n_qubits, (preset_key.upper(),))
    elif preset_key == "xy":
        obs_specs = single_site_observable_specs(n_qubits, ("X", "Y"))
    elif preset_key == "zx":
        obs_specs = single_site_observable_specs(n_qubits, ("Z", "X"))
    elif preset_key == "xyz":
        obs_specs = single_site_observable_specs(n_qubits, ("X", "Y", "Z"))
    elif preset_key == "zz_pairs":
        obs_specs = pair_observable_specs(n_qubits, ("Z",), ("Z",))
    elif preset_key == "pair_xyz":
        obs_specs = pair_observable_specs(n_qubits, ("X", "Y", "Z"), ("X", "Y", "Z"))
    elif preset_key == "rich":
        obs_specs = single_site_observable_specs(n_qubits, ("X", "Y", "Z")) + pair_observable_specs(
            n_qubits,
            ("X", "Y", "Z"),
            ("X", "Y", "Z"),
        )
    elif include_extended_pairs and preset_key == "xx_pairs":
        obs_specs = pair_observable_specs(n_qubits, ("X",), ("X",))
    elif include_extended_pairs and preset_key == "nn_pairs":
        obs_specs = nearest_neighbor_observable_specs(n_qubits, ("X", "Y", "Z"), ("X", "Y", "Z"))
    elif preset_key == "custom":
        obs_specs = []
    else:
        raise ValueError(f"Unsupported observable preset '{preset}'")

    if custom_specs:
        obs_specs.extend(custom_specs)
    return list(dict.fromkeys(obs_specs))


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
