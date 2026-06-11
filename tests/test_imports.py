def test_core_imports():
    from pyqres.core import (
        BackendLike,
        BatchReservoirProtocol,
        ConfigMapping,
        CircuitLike,
        CircuitReservoirProtocol,
        DatasetSplitProtocol,
        DatasetProtocol,
        DimensionModelProtocol,
        ExperimentProtocol,
        HamiltonianLike,
        HamiltonianSpecProtocol,
        IndexSequence,
        MemoryObservableReservoirProtocol,
        QRCReservoirProtocol,
        ObservableSpec,
        PauliTermLike,
        ReadoutSpecProtocol,
        ReadoutProtocol,
        ReservoirBuilderProtocol,
        ReservoirCompilerProtocol,
        ReservoirFactoryProtocol,
        ReservoirSpecProtocol,
        ReservoirRunResult,
        ReservoirStepResult,
        SerializableSpecProtocol,
        StepReservoirProtocol,
        StreamingReservoirProtocol,
        SupervisedDataBuilderProtocol,
        SweepProtocol,
        SweepResultProtocol,
        TaskDatasetFactoryProtocol,
        TaskRunnerProtocol,
        TimeSeriesDataBuilderProtocol,
        TransformReservoirProtocol,
    )

    assert BackendLike is not None
    assert BatchReservoirProtocol is not None
    assert CircuitLike is not None
    assert CircuitReservoirProtocol is not None
    assert QRCReservoirProtocol is not None
    assert TransformReservoirProtocol is not None
    assert DimensionModelProtocol is not None
    assert HamiltonianLike is not None
    assert HamiltonianSpecProtocol is not None
    assert IndexSequence is not None
    assert MemoryObservableReservoirProtocol is not None
    assert ObservableSpec is not None
    assert PauliTermLike is not None
    assert ReadoutSpecProtocol is not None
    assert DatasetSplitProtocol is not None
    assert DatasetProtocol is not None
    assert ExperimentProtocol is not None
    assert ReadoutProtocol is not None
    assert ReservoirBuilderProtocol is not None
    assert ReservoirCompilerProtocol is not None
    assert ReservoirFactoryProtocol is not None
    assert ReservoirSpecProtocol is not None
    assert SerializableSpecProtocol is not None
    assert StepReservoirProtocol is not None
    assert StreamingReservoirProtocol is not None
    assert SupervisedDataBuilderProtocol is not None
    assert SweepProtocol is not None
    assert SweepResultProtocol is not None
    assert TaskDatasetFactoryProtocol is not None
    assert TaskRunnerProtocol is not None
    assert TimeSeriesDataBuilderProtocol is not None
    assert ConfigMapping is not None
    assert ReservoirRunResult is not None
    assert ReservoirStepResult is not None


def test_public_imports():
    from pyqres.simulation import ExactQRCModelConfig
    from pyqres.qiskit import QRCReservoir
    from pyqres.dim import IsingReservoirModel, IsingReservoirParameters, QRCLibExactReservoirModel
    from pyqres.baselines import ESNConfig
    from pyqres import Dataset, Experiment, ReservoirSpec, Ridge

    assert ExactQRCModelConfig is not None
    assert QRCReservoir is not None
    assert IsingReservoirModel is not None
    ising_model = IsingReservoirModel(IsingReservoirParameters(n_memory=1, n_readout=1))
    assert ising_model.h0.shape == ising_model.h1.shape
    assert not hasattr(ising_model, "v")
    assert QRCLibExactReservoirModel is not None
    assert ESNConfig is not None
    assert Dataset is not None
    assert Experiment is not None
    assert ReservoirSpec is not None
    assert Ridge is not None


def test_simulation_and_dim_smoke():
    import numpy as np
    from scipy import sparse

    from pyqres.core import HamiltonianSpec, PauliTerm, ReservoirParams
    from pyqres.dim import QRCLibExactReservoirModel
    from pyqres.simulation import ChannelMapReservoir, ChannelMapReservoirConfig, ExactQRCModel, ExactQRCModelConfig

    cfg = ChannelMapReservoirConfig(n_system=1, n_ancilla=1, seed=1)
    reservoir = ChannelMapReservoir(cfg)
    assert reservoir.run(np.array([0.0, 0.1])).shape == (2, 3)

    model = QRCLibExactReservoirModel(config=cfg)
    assert model.ptm(0.0).shape == (4, 4)

    term_params = ReservoirParams.from_pauli_terms(
        n_system=1,
        n_ancilla=1,
        h0_terms=[PauliTerm(1.0, ((0, "X"),))],
        h1_terms=[(0.5, ((1, "Z"),))],
    ).generate()
    assert isinstance(term_params["H0_hamiltonian"], HamiltonianSpec)
    assert term_params["H0_matrix"] is None
    term_model = ExactQRCModel(ExactQRCModelConfig(**term_params))
    assert term_model.H0.shape == (4, 4)
    assert term_model.H1.shape == (4, 4)

    matrix_params = ReservoirParams.from_matrices(
        n_system=1,
        n_ancilla=1,
        h0_matrix=term_model.H0,
        h1_matrix=term_model.H1,
    ).generate()
    matrix_model = ExactQRCModel(ExactQRCModelConfig(**matrix_params))
    assert matrix_model.unitary(0.2).shape == (4, 4)

    sparse_params = ReservoirParams.from_matrices(
        n_system=1,
        n_ancilla=1,
        h0_matrix=sparse.csr_matrix(term_model.H0),
        h1_matrix=sparse.csr_matrix(term_model.H1),
    ).generate()
    assert isinstance(sparse_params["H0_hamiltonian"], HamiltonianSpec)
    sparse_model = ExactQRCModel(ExactQRCModelConfig(**sparse_params))
    assert sparse_model.H0.shape == (4, 4)


def test_qiskit_hamiltonian_like_inputs():
    import numpy as np
    import pytest

    qi = pytest.importorskip("qiskit.quantum_info")
    fake_provider = pytest.importorskip("qiskit.providers.fake_provider")

    from pyqres.core import ReservoirParams
    from pyqres.qiskit import QRCConfig, QRCReservoir
    from pyqres.simulation import ExactQRCModel, ExactQRCModelConfig

    ising_params = ReservoirParams.ising_type(n_system=1, n_ancilla=1, seed=2).generate()
    ising_op = ising_params["H0_hamiltonian"].to_sparse_pauli_op()
    assert isinstance(ising_op, qi.SparsePauliOp)
    spec_ising_model = ExactQRCModel(ExactQRCModelConfig(**ising_params))
    assert spec_ising_model.H0.shape == (4, 4)
    assert spec_ising_model.H1.shape == (4, 4)

    qiskit_cfg = QRCConfig(
        n_system=1,
        n_ancilla=1,
        reservoir_type="pauli_evolution",
        H0_hamiltonian=ising_params["H0_hamiltonian"],
        H1_hamiltonian=ising_params["H1_hamiltonian"],
    )
    qiskit_reservoir = QRCReservoir(qiskit_cfg)
    circuit, _, _ = qiskit_reservoir.build_streaming_circuit([0.1], measure_system=False)
    assert circuit.num_qubits == 2
    executable = qiskit_reservoir.build_executable_circuit([0.1], measure_system=False)
    assert "PauliEvolution" not in "".join(executable.count_ops().keys())

    backend = fake_provider.GenericBackendV2(
        num_qubits=2,
        basis_gates=["rx", "rz", "sx", "x", "cx", "measure", "reset"],
    )
    transpiled = qiskit_reservoir.build_executable_circuit([0.1], backend=backend, measure_system=False)
    assert transpiled.num_qubits == 2

    h0 = qi.SparsePauliOp.from_list([("XI", 1.0)])
    h1 = qi.SparsePauliOp.from_list([("IZ", 0.5)])
    params = ReservoirParams.from_matrices(
        n_system=1,
        n_ancilla=1,
        h0_matrix=h0,
        h1_matrix=h1,
    ).generate()
    assert isinstance(params["H0_hamiltonian"].to_sparse_pauli_op(), qi.SparsePauliOp)
    model = ExactQRCModel(ExactQRCModelConfig(**params))
    assert model.unitary(0.25).shape == (4, 4)


def test_generic_experiment_api_smoke():
    import numpy as np

    from pyqres import Dataset, Experiment, ReservoirSpec, Ridge, compile_reservoir

    inputs = np.linspace(-0.5, 0.5, 24)
    targets = np.roll(inputs, -1)
    dataset = Dataset.from_arrays(inputs[:-1], targets[:-1], washout=2, train=12, test=6)
    spec = ReservoirSpec(family="ising", n_system=1, n_ancilla=1, tau=0.2, seed=1)
    reservoir = compile_reservoir(spec, backend="exact")
    result = Experiment(reservoir=reservoir, dataset=dataset, readout=Ridge(l2=1e-6), metrics=["r2"]).run()

    assert result.features.shape[0] == dataset.inputs.shape[0]
    assert "test" in result.metrics
    assert "r2" in result.metrics["test"]


def test_dataset_save_load_and_config_runner(tmp_path):
    import numpy as np
    from omegaconf import OmegaConf

    from pyqres import Dataset
    from pyqres.experiments import run_experiment_from_config

    inputs = np.linspace(-0.5, 0.5, 20)
    targets = np.roll(inputs, -1)
    dataset = Dataset.from_arrays(inputs[:-1], targets[:-1], washout=2, train=10, test=5)
    npz_path = dataset.save_npz(tmp_path / "dataset.npz")

    cfg = OmegaConf.create(
        {
            "dataset": {
                "source": "npz",
                "path": str(npz_path),
                "split": {"washout": 2, "train": 10, "test": 5},
            },
            "reservoir": {
                "family": "ising",
                "n_system": 1,
                "n_ancilla": 1,
                "tau": 0.2,
                "seed": 5,
            },
            "backend": "exact",
            "readout": {"kind": "ridge", "l2": 1e-6},
            "metrics": ["r2", "mse"],
            "paths": {"output_dir": str(tmp_path / "run"), "timestamped": False},
        }
    )
    result = run_experiment_from_config(cfg)

    assert (tmp_path / "run" / "metrics.json").exists()
    assert (tmp_path / "run" / "arrays.npz").exists()
    assert "test" in result.metrics
    assert "mse" in result.metrics["test"]


def test_protocol_runtime_checks_and_dict_config(tmp_path):
    import numpy as np

    from pyqres import (
        ChannelReservoirProtocol,
        Dataset,
        DatasetProtocol,
        DimensionModelProtocol,
        ExperimentProtocol,
        HamiltonianSpecProtocol,
        MemoryObservableReservoirProtocol,
        ReadoutProtocol,
        ReadoutSpecProtocol,
        ReservoirBuilderProtocol,
        ReservoirSpec,
        ReservoirSpecProtocol,
        Ridge,
        SerializableSpecProtocol,
        StreamingReservoirProtocol,
        SupervisedDataBuilderProtocol,
        TimeSeriesDataBuilderProtocol,
        TransformReservoirProtocol,
        compile_reservoir,
        reservoir as reservoir_builder,
    )
    from pyqres.core import ReservoirParams
    from pyqres.experiments import Experiment, run_experiment_from_config

    inputs = np.linspace(-0.25, 0.25, 18)
    targets = np.roll(inputs, -1)
    dataset = Dataset.from_arrays(inputs[:-1], targets[:-1], washout=2, train=9, test=4)
    data_path = dataset.save_npz(tmp_path / "dict_dataset.npz")
    spec = ReservoirSpec(family="ising", n_system=1, n_ancilla=1, tau=0.15, seed=6)
    reservoir = compile_reservoir(spec, backend="exact")
    readout = Ridge()
    builder = reservoir_builder("ising")
    experiment = Experiment(reservoir, dataset, readout=readout)
    supervised_builder = __import__("pyqres").data.arrays(inputs[:-1], targets[:-1])
    timeseries_builder = __import__("pyqres").data.timeseries(inputs)

    assert isinstance(dataset, DatasetProtocol)
    assert isinstance(spec.readout, ReadoutSpecProtocol)
    assert isinstance(spec, SerializableSpecProtocol)
    assert isinstance(spec, ReservoirSpecProtocol)
    assert isinstance(reservoir, ChannelReservoirProtocol)
    assert isinstance(reservoir, TransformReservoirProtocol)
    assert isinstance(reservoir, StreamingReservoirProtocol)
    assert reservoir.ptm(0.0).shape == (4, 4)
    assert isinstance(readout, ReadoutProtocol)
    assert isinstance(builder, ReservoirBuilderProtocol)
    assert isinstance(experiment, ExperimentProtocol)
    assert isinstance(supervised_builder, SupervisedDataBuilderProtocol)
    assert isinstance(timeseries_builder, TimeSeriesDataBuilderProtocol)

    memory_reservoir = (
        reservoir_builder("ising")
        .memory_qubits(1)
        .readout_qubits(1)
        .observables("z", count=1)
        .backend("memory_observable")
    )
    assert isinstance(memory_reservoir, MemoryObservableReservoirProtocol)
    assert isinstance(memory_reservoir.model, DimensionModelProtocol)

    hamiltonian_spec = ReservoirParams.ising_type(
        n_system=1,
        n_ancilla=1,
    ).generate()["H0_hamiltonian"]
    assert isinstance(hamiltonian_spec, HamiltonianSpecProtocol)

    cfg = {
        "dataset": {
            "source": "npz",
            "path": str(data_path),
        },
        "reservoir": spec.to_dict(),
        "backend": "exact",
        "readout": {"kind": "ridge", "l2": 1e-6},
        "metrics": ["r2"],
        "paths": {"output_dir": str(tmp_path / "dict_run"), "timestamped": False},
    }
    result = run_experiment_from_config(cfg)

    assert (tmp_path / "dict_run" / "metrics.json").exists()
    assert "r2" in result.metrics["test"]


def test_fluent_reservoir_data_experiment_path():
    import numpy as np
    import pyqres as qres

    series = np.sin(np.linspace(0.0, 2.0, 48))
    reservoir = (
        qres.reservoir("ising")
        .memory_qubits(2)
        .readout_qubits(1)
        .input("Z", site=0, strength=1.2)
        .evolution(tau=0.2)
        .observables("rich", count=3)
        .backend("exact")
    )
    dataset = qres.data.timeseries(series, target_horizon=1).split(
        washout=4,
        train=24,
        test=12,
    )
    result = qres.Experiment(
        reservoir=reservoir,
        dataset=dataset,
        readout=qres.readout.Ridge(l2=1e-6),
        metrics=["r2", "mse"],
    ).run()

    assert result.features.shape == (47, 4)
    assert "r2" in result.metrics["test"]
    assert "mse" in result.metrics["test"]


def test_fluent_array_data_and_ancilla_features():
    import numpy as np
    import pyqres as qres

    inputs = np.linspace(-1.0, 1.0, 30)
    targets = inputs**2
    reservoir = (
        qres.reservoir("ising")
        .memory_qubits(1)
        .readout_qubits(1)
        .evolution(tau=0.1)
        .ancilla_probabilities(include_bias=True)
        .backend("exact")
    )
    dataset = qres.data.arrays(inputs, targets).split(washout=3, train=18, test=8)
    result = qres.Experiment(reservoir, dataset, readout=qres.readout.Ridge(), metrics=["mse"]).run()

    assert result.features.shape == (30, 3)
    assert "mse" in result.metrics["test"]


def test_fluent_explicit_hamiltonian_is_not_preset_bound():
    import numpy as np
    import pyqres as qres

    h0_terms = [(1.0, ((0, "X"),)), (0.3, ((0, "Z"), (1, "Z")))]
    h1_terms = [(0.5, ((1, "Z"),))]
    reservoir = (
        qres.reservoir()
        .memory_qubits(1)
        .readout_qubits(1)
        .hamiltonian(h0_terms=h0_terms, h1_terms=h1_terms)
        .ancilla_probabilities(include_bias=False)
        .backend("exact")
    )
    features = qres.transform(reservoir, np.array([0.0, 0.1, 0.2]))

    assert features.shape == (3, 2)


def test_dimension_presets_are_presets_not_core_families():
    import numpy as np
    import pyqres as qres

    reservoir = (
        qres.reservoir("random_pauli")
        .memory_qubits(2)
        .readout_qubits(1)
        .model(depth=1, seed=3)
        .observables("z", count=2)
        .backend("memory_observable")
    )
    features = qres.transform(reservoir, np.array([0.0, 0.1, 0.2]))

    assert features.shape == (3, 3)


def test_builder_can_use_existing_reservoir_object():
    import numpy as np
    import pyqres as qres

    class ExistingReservoir:
        def transform(self, inputs):
            x = np.asarray(inputs, dtype=float).reshape(-1, 1)
            return np.hstack([np.ones_like(x), x])

    reservoir = qres.reservoir().use(ExistingReservoir()).backend("exact")
    features = qres.transform(reservoir, np.array([0.2, 0.4]))

    assert features.tolist() == [[1.0, 0.2], [1.0, 0.4]]


def test_custom_qiskit_circuit_reservoir_compiles():
    import pytest
    import pyqres as qres

    from pyqres import CircuitReservoirProtocol

    qiskit = pytest.importorskip("qiskit")

    circuit = qiskit.QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    reservoir = (
        qres.reservoir()
        .memory_qubits(1)
        .readout_qubits(1)
        .circuit(circuit)
        .build()
    )
    assert isinstance(reservoir, CircuitReservoirProtocol)
    executable = reservoir.build_executable_circuit([0.1], measure_system=False)

    assert executable.num_qubits == 2


def test_qresreservoir_from_dict_api():
    import numpy as np

    from qres import qresreservoir as short_factory
    from pyqres import ReservoirBuilderProtocol, ReservoirFactoryProtocol, qresreservoir, transform

    assert short_factory is qresreservoir
    assert isinstance(qresreservoir, ReservoirFactoryProtocol)

    builder = qresreservoir.builder_from_dict(
        {
            "preset": "Ising",
            "memory_qubits": 1,
            "readout_qubits": 1,
            "input": {"axis": "Z", "site": 0, "strength": 1.2},
            "evolution": {"tau": 0.1},
            "observables": {"preset": "z", "count": 1},
            "backend": "exact",
        }
    )
    assert isinstance(builder, ReservoirBuilderProtocol)

    reservoir = qresreservoir.from_dict(
        {
            "preset": "Ising",
            "memory_qubits": 1,
            "readout_qubits": 1,
            "input": {"axis": "Z", "site": 0, "strength": 1.2},
            "evolution": {"tau": 0.1},
            "observables": {"preset": "z", "count": 1},
            "backend": "exact",
        }
    )
    features = transform(reservoir, np.array([0.0, 0.1, 0.2]))

    assert features.shape == (3, 2)


def test_qresreservoir_from_dict_explicit_hamiltonian():
    import numpy as np

    from pyqres import qresreservoir, transform

    reservoir = qresreservoir.from_dict(
        {
            "memory_qubits": 1,
            "readout_qubits": 1,
            "hamiltonian": {
                "h0_terms": [(1.0, ((0, "X"),))],
                "h1_terms": [(0.5, ((1, "Z"),))],
            },
            "readout": {"mode": "ancilla_probabilities", "include_bias": False},
            "backend": "exact",
        }
    )
    features = transform(reservoir, np.array([0.0, 0.1]))

    assert features.shape == (2, 2)
