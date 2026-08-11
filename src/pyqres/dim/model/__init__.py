"""Dimension-model families.

This package preserves the historical :mod:`pyqres.dim.model` import path
while separating the shared channel engine from each Hamiltonian family.
"""

from .base import ReservoirBase
from .ising import IsingReservoirModel, IsingReservoirParameters
from .random_pauli import RandomPauliReservoirModel, RandomPauliReservoirParameters
from .syk import SYKReservoirModel, SYKReservoirParameters

__all__ = [
    "IsingReservoirModel",
    "IsingReservoirParameters",
    "RandomPauliReservoirModel",
    "RandomPauliReservoirParameters",
    "ReservoirBase",
    "SYKReservoirModel",
    "SYKReservoirParameters",
]

# Preserve class metadata used by repr, pickle, and callers that inspect the
# original public module path.
for _public_class in (
    ReservoirBase,
    IsingReservoirModel,
    IsingReservoirParameters,
    RandomPauliReservoirModel,
    RandomPauliReservoirParameters,
    SYKReservoirModel,
    SYKReservoirParameters,
):
    _public_class.__module__ = __name__

del _public_class
