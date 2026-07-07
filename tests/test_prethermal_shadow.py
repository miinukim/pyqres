from __future__ import annotations

from itertools import product

import numpy as np
import pytest


def small_reservoir(**shadow_kwargs):
    from pyqres.prethermal_shadow import (
        GlobalFloquetConfig,
        GlobalFloquetPartialShadowReservoir,
        InputEncodingConfig,
        PartialShadowReadoutConfig,
        ReadoutResetConfig,
    )

    shadow = {"shots": 64, "seed": 12}
    shadow.update(shadow_kwargs)
    return GlobalFloquetPartialShadowReservoir(
        GlobalFloquetConfig(
            n_qubits=4,
            n_memory=2,
            n_readout=2,
            omega=14.0,
            n_cycles_per_step=1,
            seed=10,
        ),
        InputEncodingConfig(beta=0.08, seed=11),
        PartialShadowReadoutConfig(**shadow),
        ReadoutResetConfig(reset_state="zero"),
        seed_simulator=13,
    )


def test_imports_do_not_require_qiskit():
    import pyqres
    from pyqres.prethermal_shadow import GlobalFloquetConfig, PartialShadowReadoutConfig

    assert pyqres.Experiment is not None
    assert pyqres.GlobalFloquetPartialShadowReservoir is not None
    assert pyqres.GlobalFloquetInputEncodingConfig is not None
    assert GlobalFloquetConfig(n_qubits=2, n_memory=1, n_readout=1, omega=10.0, n_cycles_per_step=1)
    assert PartialShadowReadoutConfig().measurement_type == "projective"


def test_experimental_import_path_is_not_provided():
    with pytest.raises(ModuleNotFoundError):
        __import__("pyqres.experimental.prethermal_shadow")


def test_feature_count_and_names():
    res = small_reservoir(pauli_k=2, include_bias=True)
    assert len(res.get_feature_names()) == 1 + 3 * 2 + 9
    assert res.get_feature_names()[:4] == ["bias", "X_r0", "Y_r0", "Z_r0"]


def test_weak_povm_validity():
    from pyqres.prethermal_shadow.dynamics import PAULI
    from pyqres.prethermal_shadow.shadows import weak_povm_effect

    for axis in ("X", "Y", "Z"):
        e_plus = weak_povm_effect(axis, +1, 0.4)
        e_minus = weak_povm_effect(axis, -1, 0.4)
        assert np.allclose(e_plus + e_minus, PAULI["I"])
        assert np.min(np.linalg.eigvalsh(e_plus)) >= -1e-12
        assert np.min(np.linalg.eigvalsh(e_minus)) >= -1e-12


def test_weak_estimator_unbiased_single_qubit_by_enumeration():
    from pyqres.prethermal_shadow.dynamics import PAULI
    from pyqres.prethermal_shadow.shadows import outcome_probabilities

    rho = np.array([[0.65, 0.12 - 0.08j], [0.12 + 0.08j, 0.35]], dtype=complex)
    strength = 0.37
    axes = ("X", "Y", "Z")
    for target in axes:
        expected = 0.0
        for axis in axes:
            outcomes, probs = outcome_probabilities(rho, [axis], strength)
            for outcome, prob in zip(outcomes, probs):
                if axis == target:
                    expected += (1.0 / 3.0) * prob * (3.0 * outcome[0] / strength)
        exact = np.trace(PAULI[target] @ rho).real
        assert np.isclose(expected, exact, atol=1e-12)


def test_two_qubit_weak_estimator_unbiased_for_weight_two_by_enumeration():
    from pyqres.prethermal_shadow.dynamics import PAULI
    from pyqres.prethermal_shadow.shadows import outcome_probabilities

    psi = np.array([1.0, 0.2j, -0.3, 0.4], dtype=complex)
    psi = psi / np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    strength = 0.6
    target = ("X", "Z")
    expected = 0.0
    for axes in product(("X", "Y", "Z"), repeat=2):
        outcomes, probs = outcome_probabilities(rho, axes, strength)
        for outcome, prob in zip(outcomes, probs):
            if axes == target:
                expected += (1.0 / 9.0) * prob * (3.0 * outcome[0] / strength) * (3.0 * outcome[1] / strength)
    exact = np.trace(np.kron(PAULI["X"], PAULI["Z"]) @ rho).real
    assert np.isclose(expected, exact, atol=1e-12)


def test_variance_scaling_for_maximally_mixed_weight_two():
    from pyqres.prethermal_shadow.shadows import outcome_probabilities

    rho = np.eye(4, dtype=complex) / 4.0
    strength = 0.5
    second_moment = 0.0
    target = ("X", "Z")
    for axes in product(("X", "Y", "Z"), repeat=2):
        outcomes, probs = outcome_probabilities(rho, axes, strength)
        for outcome, prob in zip(outcomes, probs):
            estimate = 0.0
            if axes == target:
                estimate = (3.0 * outcome[0] / strength) * (3.0 * outcome[1] / strength)
            second_moment += (1.0 / 9.0) * prob * estimate**2
    assert np.isclose(second_moment, (3.0 / strength**2) ** 2)


def test_reset_channel_density_properties():
    res = small_reservoir(pauli_k=1, exact_expectations=True, return_shadow_estimates=False)
    features = res.step(0.2)
    rho = res.rho_memory
    assert features.shape == (1 + 3 * 2,)
    assert rho.shape == (4, 4)
    assert np.isclose(np.trace(rho), 1.0)
    assert np.allclose(rho, rho.conj().T)
    assert np.min(np.linalg.eigvalsh(rho)) >= -1e-10


def test_projected_channel_shape_and_spectrum():
    res = small_reservoir(pauli_k=1, exact_expectations=True, return_shadow_estimates=False)
    channel = res.build_projected_memory_channel(pauli_k=2)
    expected = 3 * 2 + 9
    assert channel.shape == (expected, expected)
    lams = res.memory_channel_spectrum(pauli_k=2)
    assert lams.shape == (expected,)
    assert np.all(np.isfinite(lams))


def test_reproducibility():
    inputs = np.linspace(-0.2, 0.2, 4)
    a = small_reservoir(pauli_k=1, measurement_type="weak", weak_strength=0.7, shots=32)
    b = small_reservoir(pauli_k=1, measurement_type="weak", weak_strength=0.7, shots=32)
    assert np.allclose(a.h, b.h)
    assert np.allclose(a.jz, b.jz)
    assert np.allclose(a.jxy, b.jxy)
    assert np.allclose(a.drive_coeffs, b.drive_coeffs)
    assert np.allclose(a.run(inputs), b.run(inputs))


def test_acceptance_shape():
    from pyqres.prethermal_shadow import (
        GlobalFloquetConfig,
        GlobalFloquetPartialShadowReservoir,
        InputEncodingConfig,
        PartialShadowReadoutConfig,
        ReadoutResetConfig,
    )

    res = GlobalFloquetPartialShadowReservoir(
        GlobalFloquetConfig(n_qubits=6, n_memory=4, n_readout=2, omega=12.0, n_cycles_per_step=2, seed=0),
        InputEncodingConfig(input_qubits="memory", axis="y", beta=0.12, random_beta=True, seed=1),
        PartialShadowReadoutConfig(pauli_k=2, shots=32, measurement_type="weak", weak_strength=0.5, include_bias=True, seed=2),
        ReadoutResetConfig(reset_after_measurement=True, reset_state="zero"),
        seed_simulator=3,
    )
    x = res.run(np.random.default_rng(0).normal(size=3))
    assert x.shape == (3, 1 + (3 * 2 + 9))
    assert len(res.get_feature_names()) == x.shape[1]
