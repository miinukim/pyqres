"""Unified namespace for task-agnostic quantum reservoir computing tools."""

from . import data, readout
from .core.builders import build_dimension_model, build_hamiltonian_params, compile_reservoir, transform
from .core import (
    ChannelReservoirProtocol,
    CircuitReservoirProtocol,
    ConfigMapping,
    DatasetProtocol,
    ExperimentResultProtocol,
    MetricCallable,
    QRCReservoirProtocol,
    ReadoutProtocol,
    ReservoirRunResult,
    ReservoirStepResult,
    SerializableSpecProtocol,
    StatefulReservoirProtocol,
    TransformReservoirProtocol,
)
from .core.fluent import ReservoirBuilder, reservoir
from .core.specs import ReadoutSpec, ReservoirSpec
from .experiments.datasets import Dataset, DatasetSplit
from .experiments.readout import Ridge
from .experiments.runner import Experiment, ExperimentResult, Sweep, SweepResult

__all__ = [
    "ChannelReservoirProtocol",
    "ConfigMapping",
    "CircuitReservoirProtocol",
    "Dataset",
    "DatasetProtocol",
    "DatasetSplit",
    "Experiment",
    "ExperimentResultProtocol",
    "ExperimentResult",
    "MetricCallable",
    "QRCReservoirProtocol",
    "ReadoutProtocol",
    "ReadoutSpec",
    "ReservoirRunResult",
    "ReservoirSpec",
    "ReservoirStepResult",
    "ReservoirBuilder",
    "Ridge",
    "SerializableSpecProtocol",
    "StatefulReservoirProtocol",
    "Sweep",
    "SweepResult",
    "TransformReservoirProtocol",
    "build_dimension_model",
    "build_hamiltonian_params",
    "compile_reservoir",
    "data",
    "readout",
    "reservoir",
    "transform",
]
