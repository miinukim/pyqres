"""Qiskit-compatible reservoir APIs."""

from pyqres.core import HamiltonianSpec, pauli_terms_to_sparse_pauli_op

from .config import NISQRCConfig, NoiseConfig, QRCConfig
from .reservoir import NISQReservoir, QRCReservoir

__all__ = [
    "HamiltonianSpec",
    "NISQRCConfig",
    "NISQReservoir",
    "NoiseConfig",
    "QRCConfig",
    "QRCReservoir",
    "pauli_terms_to_sparse_pauli_op",
]
