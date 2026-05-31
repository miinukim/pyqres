"""Public core API for shared reservoir protocols and configuration helpers."""

from .control import MeasurementControlConfig
from .protocols import (
    ChannelReservoirProtocol,
    CircuitReservoirProtocol,
    QRCReservoirProtocol,
    ReservoirRunResult,
    ReservoirStepResult,
)
from .reservoir_params import (
    HamiltonianSpec,
    PauliTerm,
    ReservoirParams,
    dense_hamiltonian_matrix,
    normalize_pauli_term,
    pauli_term_matrix,
    pauli_terms_matrix,
    pauli_terms_to_labels,
    pauli_terms_to_sparse_pauli_op,
)

__all__ = [
    "ChannelReservoirProtocol",
    "CircuitReservoirProtocol",
    "HamiltonianSpec",
    "MeasurementControlConfig",
    "PauliTerm",
    "QRCReservoirProtocol",
    "ReservoirParams",
    "ReservoirRunResult",
    "ReservoirStepResult",
    "dense_hamiltonian_matrix",
    "normalize_pauli_term",
    "pauli_term_matrix",
    "pauli_terms_matrix",
    "pauli_terms_to_labels",
    "pauli_terms_to_sparse_pauli_op",
]
