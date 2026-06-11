"""Unified namespace for task-agnostic quantum reservoir computing tools."""

from .builders import build_dimension_model, build_hamiltonian_params, compile_reservoir, transform
from .core import (
    ChannelReservoirProtocol,
    CircuitReservoirProtocol,
    QRCReservoirProtocol,
    ReservoirRunResult,
    ReservoirStepResult,
)
from .datasets import Dataset, DatasetSplit
from .experiment import Experiment, ExperimentResult, Sweep, SweepResult
from .readout import Ridge
from .specs import ReadoutSpec, ReservoirSpec

__all__ = [
    "ChannelReservoirProtocol",
    "CircuitReservoirProtocol",
    "Dataset",
    "DatasetSplit",
    "Experiment",
    "ExperimentResult",
    "QRCReservoirProtocol",
    "ReadoutSpec",
    "ReservoirRunResult",
    "ReservoirSpec",
    "ReservoirStepResult",
    "Ridge",
    "Sweep",
    "SweepResult",
    "build_dimension_model",
    "build_hamiltonian_params",
    "compile_reservoir",
    "transform",
]
