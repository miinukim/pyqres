def test_core_imports():
    from pyqres.core import QRCReservoirProtocol, ReservoirRunResult, ReservoirStepResult

    assert QRCReservoirProtocol is not None
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
