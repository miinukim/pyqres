"""Compatibility checks for the responsibility-based internal layout."""

import pickle


def test_legacy_modules_reexport_focused_implementations():
    from pyqres.core.protocols import DatasetProtocol, InputSequence
    from pyqres.core.protocols.experiments import (
        DatasetProtocol as FocusedDatasetProtocol,
    )
    from pyqres.core.protocols.types import InputSequence as FocusedInputSequence
    from pyqres.core.reservoir_params import HamiltonianSpec, ReservoirParams
    from pyqres.core.reservoir_params.parameters import (
        ReservoirParams as FocusedReservoirParams,
    )
    from pyqres.core.reservoir_params.specifications import (
        HamiltonianSpec as FocusedHamiltonianSpec,
    )
    from pyqres.dim.model import IsingReservoirModel, ReservoirBase
    from pyqres.dim.model.base import ReservoirBase as FocusedReservoirBase
    from pyqres.dim.model.ising import IsingReservoirModel as FocusedIsingReservoirModel
    from pyqres.prethermal_shadow.dynamics import (
        parse_pauli_terms,
        partial_trace_qubits,
    )
    from pyqres.prethermal_shadow.dynamics.operators import (
        parse_pauli_terms as FocusedParsePauliTerms,
    )
    from pyqres.prethermal_shadow.dynamics.states import (
        partial_trace_qubits as FocusedPartialTraceQubits,
    )

    assert DatasetProtocol is FocusedDatasetProtocol
    assert InputSequence is FocusedInputSequence
    assert HamiltonianSpec is FocusedHamiltonianSpec
    assert ReservoirParams is FocusedReservoirParams
    assert ReservoirBase is FocusedReservoirBase
    assert IsingReservoirModel is FocusedIsingReservoirModel
    assert parse_pauli_terms is FocusedParsePauliTerms
    assert partial_trace_qubits is FocusedPartialTraceQubits


def test_moved_public_classes_keep_legacy_module_metadata():
    from pyqres.core.protocols import DatasetProtocol, ReservoirStepResult
    from pyqres.core.reservoir_params import HamiltonianSpec, ReservoirParams
    from pyqres.dim.model import IsingReservoirModel, IsingReservoirParameters

    assert DatasetProtocol.__module__ == "pyqres.core.protocols"
    assert ReservoirStepResult.__module__ == "pyqres.core.protocols"
    assert HamiltonianSpec.__module__ == "pyqres.core.reservoir_params"
    assert ReservoirParams.__module__ == "pyqres.core.reservoir_params"
    assert IsingReservoirModel.__module__ == "pyqres.dim.model"
    assert IsingReservoirParameters.__module__ == "pyqres.dim.model"


def test_legacy_dataclass_pickle_paths_round_trip():
    from pyqres.core.protocols import ReservoirStepResult
    from pyqres.core.reservoir_params import HamiltonianSpec
    from pyqres.dim.model import IsingReservoirParameters

    values = (
        ReservoirStepResult(features=[]),
        HamiltonianSpec.from_pauli_terms(1, [(1.0, ((0, "Z"),))]),
        IsingReservoirParameters(n_memory=1, n_readout=1),
    )

    for value in values:
        restored = pickle.loads(pickle.dumps(value))
        assert type(restored) is type(value)
        assert restored == value
