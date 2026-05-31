"""Dense simulation and channel-map reservoir APIs.

This is the primary namespace for small-system dense simulation backends. Some
classes retain "Exact" in their names to distinguish exact dense simulation from
finite-shot trajectory or Qiskit execution.
"""

from pyqres.core import MeasurementControlConfig, ReservoirParams

from .channel_map import ChannelMapReservoir, ChannelMapReservoirConfig
from .exact_qrc import ExactQRCModel, ExactQRCModelConfig
from .hardware import HardwareTrajectoryReservoir, HardwareTrajectoryReservoirConfig

__all__ = [
    "ChannelMapReservoir",
    "ChannelMapReservoirConfig",
    "ExactQRCModel",
    "ExactQRCModelConfig",
    "HardwareTrajectoryReservoir",
    "HardwareTrajectoryReservoirConfig",
    "MeasurementControlConfig",
    "ReservoirParams",
]
