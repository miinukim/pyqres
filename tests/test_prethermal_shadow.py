from __future__ import annotations

import numpy as np
import pytest


def small_config(**kwargs):
    from pyqres.experimental.prethermal_shadow import (
        InputWriteConfig,
        PrethermalFloquetConfig,
        PrethermalShadowConfig,
        ShadowReadoutConfig,
        TransducerConfig,
    )

    base = {
        "n_readout": 2,
        "floquet": PrethermalFloquetConfig(n_memory=2, n_floquet=1, tau=0.1, seed=5),
        "input_write": InputWriteConfig(beta=0.05),
        "transducer": TransducerConfig(tau_c=0.02, seed=6),
        "shadow": ShadowReadoutConfig(pauli_k=2, shots=8, seed=7),
        "simulator_method": "density_matrix",
        "seed_simulator": 11,
    }
    base.update(kwargs)
    return PrethermalShadowConfig(**base)


def test_imports_do_not_require_qiskit_execution():
    import pyqres
    from pyqres.experimental.prethermal_shadow.config import PrethermalFloquetConfig
    from pyqres.experimental.prethermal_shadow.shadows import generate_pauli_labels

    assert pyqres.Experiment is not None
    assert PrethermalFloquetConfig(n_memory=1).n_memory == 1
    assert len(generate_pauli_labels(2, 1)) == 6


def test_feature_label_count():
    from pyqres.experimental.prethermal_shadow.shadows import feature_label_strings

    labels = feature_label_strings(n_readout=3, pauli_k=2, include_bias=True)
    assert len(labels) == 1 + 3 * 3 + 9 * 3
    assert labels[:4] == ["bias", "X0", "Y0", "Z0"]


def test_snapshot_estimator_known_values():
    from pyqres.experimental.prethermal_shadow.shadows import estimate_shadow_features, generate_pauli_labels

    schedules = np.array(
        [
            [["Z", "X"], ["Z", "X"]],
            [["Z", "Y"], ["X", "X"]],
            [["X", "X"], ["Z", "X"]],
        ],
        dtype="U1",
    )
    outcomes = np.array(
        [
            [[0, 1], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 0], [0, 0]],
        ],
        dtype=np.int8,
    )
    labels = generate_pauli_labels(2, 2)
    features = estimate_shadow_features(schedules, outcomes, labels, include_bias=True)
    label_to_col = {"bias": 0}
    for idx, label in enumerate(labels, start=1):
        label_to_col["*".join(f"{p}{q}" for q, p in label)] = idx

    assert np.allclose(features[:, label_to_col["Z0"]], [2.0, 0.0])
    assert np.allclose(features[:, label_to_col["X1"]], [0.0, 1.0])
    assert np.allclose(features[:, label_to_col["Z0*X1"]], [-3.0, 0.0])


def test_custom_feature_builder_hook():
    from pyqres.experimental.prethermal_shadow import PrethermalShadowReservoir

    def sampler(shots, n_steps, n_readout, bases, seed):
        return np.full((shots, n_steps, n_readout), "Z", dtype="U1")

    def executor(inputs, schedule, cfg, shots):
        return {"0" * (len(inputs) * cfg.base.n_readout): shots}

    def builder(schedules, outcomes, labels, include_bias):
        return np.full((schedules.shape[1], 2), 42.0)

    reservoir = PrethermalShadowReservoir(
        small_config(),
        basis_sampler=sampler,
        schedule_executor=executor,
        feature_builder=builder,
    )
    assert np.all(reservoir.run_stream([0.0, 0.1]) == 42.0)


def test_duplicate_basis_schedules_are_grouped():
    from pyqres.experimental.prethermal_shadow import PrethermalShadowReservoir, ShadowReadoutConfig

    seen_group_sizes = []

    def sampler(shots, n_steps, n_readout, bases, seed):
        schedules = np.full((shots, n_steps, n_readout), "Z", dtype="U1")
        schedules[-1, :, :] = "X"
        return schedules

    def executor(inputs, schedule, cfg, shots):
        seen_group_sizes.append(shots)
        return {"0" * (len(inputs) * cfg.base.n_readout): shots}

    reservoir = PrethermalShadowReservoir(
        small_config(shadow=ShadowReadoutConfig(pauli_k=1, shots=5)),
        basis_sampler=sampler,
        schedule_executor=executor,
    )
    reservoir.run_stream([0.0, 0.1])
    assert sorted(seen_group_sizes) == [1, 4]


def test_zero_transducer_with_mock_executor_shape():
    from pyqres.experimental.prethermal_shadow import PrethermalFloquetConfig, PrethermalShadowConfig, PrethermalShadowReservoir, ShadowReadoutConfig

    def sampler(shots, n_steps, n_readout, bases, seed):
        return np.full((shots, n_steps, n_readout), "Z", dtype="U1")

    def executor(inputs, schedule, cfg, shots):
        return {"0" * (len(inputs) * cfg.base.n_readout): shots}

    reservoir = PrethermalShadowReservoir(
        PrethermalShadowConfig(
            n_readout=1,
            floquet=PrethermalFloquetConfig(n_memory=1, n_floquet=1),
            transducer=None,
            shadow=ShadowReadoutConfig(pauli_k=1, shots=4),
        ),
        basis_sampler=sampler,
        schedule_executor=executor,
    )
    features = reservoir.run_stream([0.0, 0.2, 0.4])
    assert features.shape == (3, 4)


def test_circuit_measures_and_resets_only_readout():
    pytest.importorskip("qiskit")

    from pyqres.experimental.prethermal_shadow import PrethermalShadowReservoir

    reservoir = PrethermalShadowReservoir(small_config())
    schedule = np.full((2, 2), "Z", dtype="U1")
    circuit = reservoir.build_streaming_circuit([0.0, 0.1], schedule)
    memory = set(range(reservoir.cfg.base.floquet.n_memory))
    readout = set(range(reservoir.cfg.base.floquet.n_memory, reservoir.cfg.base.floquet.n_memory + reservoir.cfg.base.n_readout))

    for instruction in circuit.data:
        if instruction.operation.name in {"measure", "reset"}:
            touched = {circuit.find_bit(qubit).index for qubit in instruction.qubits}
            assert touched <= readout
            assert not touched & memory


def test_deterministic_aer_execution_and_experiment_smoke(tmp_path):
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")

    import pyqres as qres
    from pyqres.experimental.prethermal_shadow import PrethermalFloquetConfig, PrethermalShadowReservoir, ShadowReadoutConfig

    cfg = small_config(
        n_readout=1,
        shadow=ShadowReadoutConfig(pauli_k=1, shots=6, seed=17),
        floquet=PrethermalFloquetConfig(n_memory=1, n_floquet=1, tau=0.05, seed=13),
        transducer=None,
        simulator_method="density_matrix",
        seed_simulator=19,
    )
    inputs = np.linspace(0.0, 0.2, 5)
    reservoir_a = PrethermalShadowReservoir(cfg)
    reservoir_b = PrethermalShadowReservoir(cfg)
    xa = reservoir_a.run_stream(inputs)
    xb = reservoir_b.run_stream(inputs)
    assert np.allclose(xa, xb)
    assert np.array_equal(reservoir_a.last_basis_schedules, reservoir_b.last_basis_schedules)

    dataset = qres.data.arrays(inputs, np.roll(inputs, -1)).split(washout=1, train=2, test=2)
    result = qres.Experiment(reservoir_a, dataset, readout=qres.Ridge(l2=1e-6), metrics=["mse"]).run()
    out = result.save(tmp_path / "run")
    assert (out / "metrics.json").exists()
    assert result.features.shape[0] == inputs.shape[0]
