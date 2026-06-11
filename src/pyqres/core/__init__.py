"""Public core API for shared reservoir protocols and configuration helpers."""

from .control import MeasurementControlConfig
from .builders import build_dimension_model, build_hamiltonian_params, compile_reservoir, transform
from .fluent import ReservoirBuilder, reservoir
from .protocols import (
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
from .specs import ReadoutSpec, ReservoirSpec

__all__ = [
    "ChannelReservoirProtocol",
    "CircuitReservoirProtocol",
    "ConfigMapping",
    "DatasetProtocol",
    "ExperimentResultProtocol",
    "HamiltonianSpec",
    "MeasurementControlConfig",
    "MetricCallable",
    "PauliTerm",
    "QRCReservoirProtocol",
    "ReadoutSpec",
    "ReadoutProtocol",
    "ReservoirParams",
    "ReservoirRunResult",
    "ReservoirSpec",
    "ReservoirStepResult",
    "ReservoirBuilder",
    "SerializableSpecProtocol",
    "StatefulReservoirProtocol",
    "TransformReservoirProtocol",
    "build_dimension_model",
    "build_hamiltonian_params",
    "compile_reservoir",
    "dense_hamiltonian_matrix",
    "normalize_pauli_term",
    "pauli_term_matrix",
    "pauli_terms_matrix",
    "pauli_terms_to_labels",
    "pauli_terms_to_sparse_pauli_op",
    "reservoir",
    "transform",
]
