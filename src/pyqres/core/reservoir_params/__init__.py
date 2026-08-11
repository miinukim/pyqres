"""Backend-neutral reservoir Hamiltonian construction.

This compatibility package preserves pyqres.core.reservoir_params while
separating operator conversion, Hamiltonian specifications, and presets.
"""

from .operators import (
    PAULI_1Q,
    PauliTerm,
    dense_hamiltonian_matrix,
    normalize_pauli_term,
    pauli_term_matrix,
    pauli_terms_matrix,
    pauli_terms_to_labels,
    pauli_terms_to_sparse_pauli_op,
)
from .parameters import ReservoirParams
from .specifications import HamiltonianSpec

__all__ = [
    "PAULI_1Q",
    "HamiltonianSpec",
    "PauliTerm",
    "ReservoirParams",
    "dense_hamiltonian_matrix",
    "normalize_pauli_term",
    "pauli_term_matrix",
    "pauli_terms_matrix",
    "pauli_terms_to_labels",
    "pauli_terms_to_sparse_pauli_op",
]

# Keep metadata stable for callers and old pickle payloads.
for _public_name in __all__:
    _public_object = globals()[_public_name]
    if hasattr(_public_object, "__code__") or isinstance(_public_object, type):
        _public_object.__module__ = __name__

del _public_name, _public_object
