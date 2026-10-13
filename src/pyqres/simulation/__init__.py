"""Dense simulation and channel-map reservoir APIs.

This is the primary namespace for small-system dense simulation backends. Some
classes retain "Exact" in their names to distinguish exact dense simulation from
finite-shot trajectory or Qiskit execution.
"""

from pyqres.core import (
    HamiltonianSpec,
    MeasurementControlConfig,
    PauliTerm,
    ReservoirParams,
)

from .channel_map import (
    ChannelMapReservoir,
    ChannelMapReservoirConfig,
    ObservableChannelMapReservoir,
    ObservableChannelMapReservoirConfig,
    parse_readout_observable,
)
from .exact_qrc import ExactQRCModel, ExactQRCModelConfig
from .hardware import HardwareTrajectoryReservoir, HardwareTrajectoryReservoirConfig
from .hu_ising import (
    HuIsingConfig,
    HuIsingReservoir,
    IsingReservoir,
    IsingReservoirConfig,
)
from .scrambling_circuit import ScramblingCircuitConfig, ScramblingCircuitReservoir
from .syk import SYKReservoir, SYKReservoirConfig

__all__ = [
    "ChannelMapReservoir",
    "ChannelMapReservoirConfig",
    "ExactQRCModel",
    "ExactQRCModelConfig",
    "HamiltonianSpec",
    "HardwareTrajectoryReservoir",
    "HardwareTrajectoryReservoirConfig",
    "HuIsingConfig",
    "HuIsingReservoir",
    "IsingReservoir",
    "IsingReservoirConfig",
    "MeasurementControlConfig",
    "ObservableChannelMapReservoir",
    "ObservableChannelMapReservoirConfig",
    "PauliTerm",
    "ReservoirParams",
    "SYKReservoir",
    "SYKReservoirConfig",
    "ScramblingCircuitConfig",
    "ScramblingCircuitReservoir",
    "parse_readout_observable",
]
