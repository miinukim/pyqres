"""Pauli operators and symbolic parsing for prethermal dynamics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

PAULI = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

PauliPlacements = tuple[tuple[int, str], ...]
PauliTerm = tuple[float, PauliPlacements]


def kron_all(ops: Sequence[np.ndarray]) -> np.ndarray:
    out = np.array([[1.0 + 0.0j]])
    for op in ops:
        out = np.kron(out, op)
    return out


def pauli_string(n_qubits: int, placements: Sequence[tuple[int, str]]) -> np.ndarray:
    ops = [PAULI["I"] for _ in range(int(n_qubits))]
    for site, pauli in placements:
        index = int(site)
        if index < 0 or index >= int(n_qubits):
            raise ValueError(
                f"Pauli placement qubit {index} is outside [0, {int(n_qubits) - 1}]."
            )
        key = str(pauli).upper()
        if key not in PAULI or key == "I":
            raise ValueError(f"unsupported Pauli {pauli!r}")
        ops[index] = PAULI[key]
    return kron_all(ops)


def parse_pauli_placements(spec: str) -> PauliPlacements:
    """Parse one Pauli string like ``0:Z,2:X`` into placements."""

    text = str(spec).strip()
    if not text or text.upper() == "I":
        return ()
    placements = []
    for part in text.split(","):
        site_text, pauli_text = part.strip().split(":", 1)
        pauli = pauli_text.strip().upper()
        if pauli not in {"X", "Y", "Z"}:
            raise ValueError(f"unsupported Pauli {pauli!r}")
        placements.append((int(site_text), pauli))
    return tuple(placements)


def parse_pauli_terms(spec: str, *, normalize: bool = True) -> tuple[PauliTerm, ...]:
    """Parse a weighted sum of Pauli strings without allocating dense matrices."""

    text = str(spec).strip()
    raw_terms = (
        ("I",)
        if not text or text.upper() == "I"
        else text.replace("-", "+-").split("+")
    )
    combined: dict[PauliPlacements, float] = {}
    for raw_term in raw_terms:
        term = raw_term.strip()
        if not term:
            continue
        if "*" in term:
            coeff_text, pauli_text = term.split("*", 1)
            coeff = float(coeff_text.strip())
        else:
            coeff = 1.0
            pauli_text = term
        placements = parse_pauli_placements(pauli_text)
        by_site = {int(site): str(pauli).upper() for site, pauli in placements}
        canonical = tuple(sorted(by_site.items()))
        combined[canonical] = combined.get(canonical, 0.0) + coeff

    terms = [
        (coeff, placements)
        for placements, coeff in combined.items()
        if abs(coeff) > 1e-15
    ]
    if normalize and terms:
        norm = float(np.sqrt(sum(float(coeff) ** 2 for coeff, _ in terms)))
        terms = [(float(coeff) / norm, placements) for coeff, placements in terms]
    return tuple(terms)


def pauli_terms_matrix(n_qubits: int, terms: Sequence[PauliTerm]) -> np.ndarray:
    """Construct a dense Hermitian matrix from symbolic Pauli terms."""

    dim = 2 ** int(n_qubits)
    out = np.zeros((dim, dim), dtype=complex)
    for coeff, placements in terms:
        out += float(coeff) * pauli_string(int(n_qubits), placements)
    return 0.5 * (out + out.conj().T)


def parse_pauli_operator(
    n_qubits: int,
    spec: str,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Parse sums of Pauli strings into a dense Hermitian operator.

    Supported examples include ``0:Y``, ``0:Z,1:Z`` and
    ``0.5*0:X + -1.2*2:Z``.
    """

    return pauli_terms_matrix(
        int(n_qubits),
        parse_pauli_terms(spec, normalize=normalize),
    )


def validate_axis(axis: str, name: str) -> str:
    """Normalize and validate a single-qubit Pauli axis."""

    out = str(axis).upper()
    if out not in {"X", "Y", "Z"}:
        raise ValueError(f"{name} must be x, y, or z.")
    return out
