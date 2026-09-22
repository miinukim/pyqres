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


def test_asymmetric_anchor_uses_boundary_bond_and_directed_drive():
    from pyqres.prethermal_shadow import build_asymmetric_anchor_reservoir

    reservoir, metadata = build_asymmetric_anchor_reservoir(
        n_memory=2,
        n_readout=2,
        omega=15.0,
        n_cycles_per_step=1,
        interface_jxy=0.025,
        interface_jz=1.2,
        drive_amplitude=0.7,
    )

    assert (metadata.anchor_qubit, metadata.readout_qubit) == (1, 2)
    assert reservoir.edges[metadata.interface_edge_index] == (1, 2)
    assert reservoir.jxy[metadata.interface_edge_index] == pytest.approx(0.025)
    assert reservoir.jz[metadata.interface_edge_index] == pytest.approx(1.2)
    assert reservoir.drive_coeffs[metadata.anchor_qubit] == pytest.approx(0.0)
    assert reservoir.drive_coeffs[metadata.readout_qubit] == pytest.approx(0.7)


def test_anchor_mode_uses_traceless_observable_quotient():
    from pyqres.prethermal_shadow import (
        build_asymmetric_anchor_reservoir,
        slow_mode_diagnostics,
    )

    reservoir, metadata = build_asymmetric_anchor_reservoir(
        n_memory=1,
        n_readout=1,
        omega=15.0,
        n_cycles_per_step=1,
        interface_jxy=0.01,
    )
    diagnostics = slow_mode_diagnostics(reservoir, metadata)

    assert diagnostics["anchor_overlap"] > 0.999
    assert diagnostics["channel_gap"] > 0.0
    assert diagnostics["channel_gap"] < 1e-3
    assert diagnostics["normalized_visibility"] > 0.1
    assert diagnostics["eigen_residual"] < 1e-10


def test_weak_interface_has_linear_dressing_and_quadratic_gap_locally():
    from pyqres.prethermal_shadow import (
        build_asymmetric_anchor_reservoir,
        slow_mode_diagnostics,
    )

    values = []
    for coupling in (0.005, 0.01):
        reservoir, metadata = build_asymmetric_anchor_reservoir(
            n_memory=1,
            n_readout=1,
            omega=15.0,
            n_cycles_per_step=1,
            interface_jxy=coupling,
        )
        values.append(slow_mode_diagnostics(reservoir, metadata))

    dressing_ratio = values[1]["dressing_norm"] / values[0]["dressing_norm"]
    gap_ratio = values[1]["channel_gap"] / values[0]["channel_gap"]
    assert dressing_ratio == pytest.approx(2.0, rel=0.02)
    assert gap_ratio == pytest.approx(4.0, rel=0.02)


def test_pauli_weight_distribution_recovers_single_site_weight():
    from pyqres.prethermal_shadow import pauli_weight_distribution
    from pyqres.prethermal_shadow.dynamics import pauli_string

    distribution = pauli_weight_distribution(
        pauli_string(2, [(0, "X")]),
        n_qubits=2,
    )

    assert np.allclose(distribution, [0.0, 1.0, 0.0])


def test_first_order_floquet_error_decreases_with_frequency():
    from pyqres.prethermal_shadow import (
        build_asymmetric_anchor_reservoir,
        floquet_effective_diagnostics,
    )

    errors = []
    for omega in (8.0, 30.0):
        reservoir, metadata = build_asymmetric_anchor_reservoir(
            n_memory=1,
            n_readout=1,
            omega=omega,
            n_cycles_per_step=1,
            interface_jxy=0.05,
        )
        diagnostics = floquet_effective_diagnostics(
            reservoir,
            {"Z_m": ((metadata.anchor_qubit, "Z"),)},
        )
        errors.append(diagnostics["unitary_error"]["frobenius"])

    assert errors[1] < errors[0]


def test_anchor_branch_tracks_from_conserved_all_to_all_limit():
    from pyqres.prethermal_shadow import (
        PrethermalShadowDimensionModel,
        build_asymmetric_anchor_reservoir,
        identify_anchor_mode,
        slow_mode_visibility,
    )

    baseline, baseline_metadata = build_asymmetric_anchor_reservoir(
        n_memory=2,
        n_readout=2,
        omega=15.0,
        n_cycles_per_step=1,
        interface_jxy=0.0,
        topology="all_to_all",
        randomize_bulk=True,
        seed=11,
    )
    reference = identify_anchor_mode(
        PrethermalShadowDimensionModel(baseline),
        anchor_memory_index=baseline_metadata.anchor_memory_index,
    )
    reservoir, metadata = build_asymmetric_anchor_reservoir(
        n_memory=2,
        n_readout=2,
        omega=15.0,
        n_cycles_per_step=1,
        interface_jxy=0.02,
        topology="all_to_all",
        randomize_bulk=True,
        seed=11,
    )
    model = PrethermalShadowDimensionModel(reservoir)
    mode = identify_anchor_mode(
        model,
        anchor_memory_index=metadata.anchor_memory_index,
        reference_operator=reference.operator,
    )
    visibility = slow_mode_visibility(
        model,
        mode,
        readout_local_index=metadata.readout_local_index,
    )

    assert mode.anchor_overlap > 0.99
    assert mode.spectral_isolation > 0.0
    assert np.isfinite(mode.eigenbasis_condition)
    assert visibility["biorthogonal_projection_abs"] > 0.0


def test_fixed_degree_interface_preserves_anchor_local_environment():
    from pyqres.prethermal_shadow import build_asymmetric_anchor_reservoir

    reservoir, metadata = build_asymmetric_anchor_reservoir(
        n_memory=2,
        n_readout=3,
        omega=30.0,
        n_cycles_per_step=1,
        interface_jxy=0.05,
        topology="all_to_all",
        isolate_anchor_diagonal=True,
        interface_readout_bulk_degree=1,
    )
    active_readout_bulk_bonds = 0
    for index, edge in enumerate(reservoir.edges):
        if metadata.anchor_qubit in edge and index != metadata.interface_edge_index:
            assert reservoir.jz[index] == pytest.approx(0.0)
            assert reservoir.jxy[index] == pytest.approx(0.0)
        if metadata.readout_qubit in edge and index != metadata.interface_edge_index:
            if abs(reservoir.jz[index]) > 0.0 or abs(reservoir.jxy[index]) > 0.0:
                active_readout_bulk_bonds += 1
    assert active_readout_bulk_bonds == 1


def test_impulse_response_returns_exact_and_modal_delay_curves():
    from pyqres.prethermal_shadow import (
        build_asymmetric_anchor_reservoir,
        impulse_response_diagnostics,
    )

    reservoir, metadata = build_asymmetric_anchor_reservoir(
        n_memory=1,
        n_readout=1,
        omega=15.0,
        n_cycles_per_step=1,
        interface_jxy=0.02,
    )
    result = impulse_response_diagnostics(
        reservoir,
        metadata,
        input_amplitude=0.01,
        max_delay=5,
    )

    assert result["delays"].tolist() == [1, 2, 3, 4, 5]
    assert result["exact_signal"].shape == (5,)
    assert result["modal_signal"].shape == (5,)
    assert result["source_amplitude_abs"] > 0.0
    assert np.isfinite(result["tail_relative_modal_error"])


def test_experimental_import_path_is_not_provided():
    with pytest.raises(ModuleNotFoundError):
        __import__("pyqres.experimental")


def test_prethermal_shadow_builds_through_standard_factory():
    import pyqres as qres

    reservoir = qres.qresreservoir.from_dict(
        {
            "preset": "prethermal_shadow",
            "memory_qubits": 2,
            "readout_qubits": 2,
            "backend": "exact",
            "floquet": {
                "omega": 14.0,
                "n_cycles_per_step": 1,
                "seed": 10,
            },
            "encoding": {
                "mode": "prethermal_shadow",
                "input_qubits": "memory",
                "axis": "y",
                "scale": 0.08,
                "seed": 11,
            },
            "shadow": {
                "pauli_k": 2,
                "shots": 64,
                "seed": 12,
            },
            "reset": {"reset_state": "zero"},
        }
    )

    features = qres.run(reservoir, np.array([0.0, 0.1]))

    assert features.shape == (2, 1 + 3 * 2 + 9)
    assert reservoir.get_feature_names()[:4] == ["bias", "X_r0", "Y_r0", "Z_r0"]


def test_standard_factory_accepts_manual_readout_indices():
    import pyqres as qres

    reservoir = qres.qresreservoir.from_dict(
        {
            "preset": "prethermal_shadow",
            "memory_qubits": 2,
            "readout_qubits": 2,
            "backend": "exact",
            "floquet": {
                "omega": 14.0,
                "n_cycles_per_step": 1,
                "readout_qubits": [1, 3],
            },
            "shadow": {
                "pauli_k": 1,
                "exact_expectations": True,
                "return_shadow_estimates": False,
            },
        }
    )

    assert reservoir.memory_qubits == (0, 2)
    assert reservoir.readout_qubits == (1, 3)


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


def test_paper_reset_covariance_and_interface_jump_match_supplement():
    from pyqres.prethermal_shadow import paper_interface_gksl_diagnostics

    diagnostics = paper_interface_gksl_diagnostics(
        (0.0125, 0.025, 0.05),
        interface_jxy=0.08,
        directed_eta=0.4,
        interface_jz=0.7,
        reverse_eta=0.03,
    )

    expected_covariance = np.asarray(
        [[1.0, 1.0j, 0.0], [-1.0j, 1.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=complex,
    )
    assert np.allclose(diagnostics["covariance_matrix"], expected_covariance)
    assert np.allclose(diagnostics["covariance_eigenvalues"], (0.0, 0.0, 2.0))
    assert diagnostics["jump_formula_relative_error"] < 1e-14


def test_paper_finite_collision_gksl_error_is_third_order_and_cptp():
    from pyqres.prethermal_shadow import paper_interface_gksl_diagnostics

    diagnostics = paper_interface_gksl_diagnostics(
        (0.00625, 0.0125, 0.025, 0.05),
        interface_jxy=0.08,
        directed_eta=0.4,
        interface_jz=0.7,
        reverse_eta=0.03,
        memory_field=0.6,
        readout_field=0.2,
    )

    assert diagnostics["gksl_error_scaling_exponent"] == pytest.approx(3.0, abs=0.02)
    assert diagnostics["second_order_error_scaling_exponent"] == pytest.approx(
        3.0, abs=0.02
    )
    assert diagnostics["literal_main_text_error_scaling_exponent"] == pytest.approx(
        2.0, abs=0.03
    )
    for record in diagnostics["records"]:
        assert record["exact_tp_residual"] < 1e-12
        assert record["gksl_tp_residual"] < 1e-12
        assert record["exact_choi_min_eigenvalue"] > -1e-12
        assert record["gksl_choi_min_eigenvalue"] > -1e-12


def test_stm_reservoir_step_gksl_diagnostic_uses_full_cycle_block():
    from pyqres.prethermal_shadow import reservoir_step_gksl_diagnostics

    reservoir = small_reservoir(
        pauli_k=1,
        exact_expectations=True,
        return_shadow_estimates=False,
    )
    diagnostics = reservoir_step_gksl_diagnostics(
        reservoir,
        duration_fractions=(1.0 / 64.0, 1.0 / 32.0, 1.0 / 16.0, 1.0 / 8.0),
    )

    assert diagnostics["configured_cycles_per_step"] == 1
    assert diagnostics["practical_collision_duration"] == pytest.approx(
        2.0 * np.pi / reservoir.floquet_config.omega
    )
    assert diagnostics["small_duration_error_exponent"] == pytest.approx(
        3.0, abs=0.15
    )
    assert len(diagnostics["cycle_records"]) == 1
    for record in diagnostics["fraction_records"]:
        assert record["gksl_tp_residual"] < 1e-11
        assert record["gksl_choi_min_eigenvalue"] > -1e-10


def test_projected_channel_shape_and_spectrum():
    res = small_reservoir(pauli_k=1, exact_expectations=True, return_shadow_estimates=False)
    channel = res.build_projected_memory_channel(pauli_k=2)
    expected = 3 * 2 + 9
    assert channel.shape == (expected, expected)
    lams = res.memory_channel_spectrum(pauli_k=2)
    assert lams.shape == (expected,)
    assert np.all(np.isfinite(lams))


def test_dimension_model_matches_prethermal_memory_channel():
    from pyqres.prethermal_shadow import PrethermalShadowDimensionModel

    reservoir = small_reservoir(pauli_k=1, exact_expectations=True, return_shadow_estimates=False)
    model = PrethermalShadowDimensionModel(reservoir)
    rng = np.random.default_rng(41)
    state = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    state = state @ state.conj().T
    state /= np.trace(state)

    expected = reservoir._memory_channel(state, u_bar=0.17)
    actual = model.channel(0.17, state)

    assert np.allclose(actual, expected, atol=1e-11)


def test_induced_feature_observables_match_exact_stm_features():
    from pyqres.prethermal_shadow import PrethermalShadowDimensionModel

    reservoir = small_reservoir(pauli_k=1, exact_expectations=True, return_shadow_estimates=False)
    model = PrethermalShadowDimensionModel(reservoir)
    rng = np.random.default_rng(42)
    state = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    state = state @ state.conj().T
    state /= np.trace(state)
    expansion_point = -0.12

    rho_pre = reservoir._pre_measurement_state(state, expansion_point)
    expected = reservoir.exact_features_from_state(rho_pre)[1:]
    observables = model.feature_observables(expansion_point)
    actual = np.asarray([np.trace(observable @ state).real for observable in observables])

    assert model.feature_observable_names() == reservoir.get_feature_names()[1:]
    assert np.allclose(actual, expected, atol=1e-11)


def test_prethermal_volterra_and_isotropy_metrics_smoke():
    from pyqres.dim import VolterraAnalyzer, ambient_readout_matrix, compressed_visibility_diagnostics
    from pyqres.prethermal_shadow import PrethermalShadowDimensionModel

    reservoir = small_reservoir(pauli_k=1, exact_expectations=True, return_shadow_estimates=False)
    model = PrethermalShadowDimensionModel(reservoir)
    observables = model.feature_observables(0.0)[:3]
    result = VolterraAnalyzer(
        model,
        observables=observables,
        max_order=1,
        lag_horizon=1,
        max_basis_size=12,
    ).analyze(n_shots=64)
    diagnostics = compressed_visibility_diagnostics(
        result.latent_basis_matrix,
        ambient_readout_matrix(observables),
    )

    assert 0 <= result.ovd <= result.vvr <= result.latent_dim
    assert diagnostics.s_gamma == result.latent_dim
    assert diagnostics.visibility_angle_deg.shape == (result.latent_dim,)
    assert np.all(np.isfinite(diagnostics.visibility_angle_deg))


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


def test_arbitrary_readout_partition_preserves_subsystem_states():
    from pyqres.prethermal_shadow.dynamics import combine_subsystem_states, partial_trace_qubits

    rng = np.random.default_rng(51)
    rho_memory = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    rho_memory = rho_memory @ rho_memory.conj().T
    rho_memory /= np.trace(rho_memory)
    rho_readout = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    rho_readout = rho_readout @ rho_readout.conj().T
    rho_readout /= np.trace(rho_readout)

    memory_qubits = (0, 2)
    readout_qubits = (3, 1)
    joint = combine_subsystem_states(rho_memory, rho_readout, memory_qubits, readout_qubits)

    assert np.allclose(partial_trace_qubits(joint, 4, memory_qubits), rho_memory)
    assert np.allclose(partial_trace_qubits(joint, 4, readout_qubits), rho_readout)


def test_noncontiguous_readout_reservoir_and_dimension_channel_agree():
    from pyqres.prethermal_shadow import (
        GlobalFloquetConfig,
        GlobalFloquetPartialShadowReservoir,
        InputEncodingConfig,
        PartialShadowReadoutConfig,
        PrethermalShadowDimensionModel,
    )

    reservoir = GlobalFloquetPartialShadowReservoir(
        GlobalFloquetConfig(
            n_qubits=4,
            n_memory=2,
            n_readout=2,
            omega=14.0,
            n_cycles_per_step=1,
            readout_qubits=(3, 1),
            seed=52,
        ),
        InputEncodingConfig(input_qubits="memory", beta=0.08, random_beta=False),
        PartialShadowReadoutConfig(
            pauli_k=1,
            exact_expectations=True,
            return_shadow_estimates=False,
        ),
    )
    assert reservoir.memory_qubits == (0, 2)
    assert reservoir.readout_qubits == (3, 1)
    assert reservoir.input_qubits == (0, 2)
    assert reservoir.run([0.1, -0.2]).shape == (2, 7)

    rng = np.random.default_rng(53)
    state = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    state = state @ state.conj().T
    state /= np.trace(state)
    model = PrethermalShadowDimensionModel(reservoir)
    assert np.allclose(model.channel(0.17, state), reservoir._memory_channel(state, 0.17), atol=1e-11)
    rho_pre = reservoir._pre_measurement_state(state, 0.17)
    expected_features = reservoir.exact_features_from_state(rho_pre)[1:]
    feature_observables = model.feature_observables(0.17)
    actual_features = np.asarray([np.trace(observable @ state).real for observable in feature_observables])
    assert np.allclose(actual_features, expected_features, atol=1e-11)


@pytest.mark.parametrize("readout_qubits", [(1,), (1, 1), (-1, 2), (1, 4)])
def test_invalid_manual_readout_indices_are_rejected(readout_qubits):
    from pyqres.prethermal_shadow import GlobalFloquetConfig, GlobalFloquetPartialShadowReservoir

    config = GlobalFloquetConfig(
        n_qubits=4,
        n_memory=2,
        n_readout=2,
        omega=14.0,
        n_cycles_per_step=1,
        readout_qubits=readout_qubits,
    )
    with pytest.raises(ValueError, match="readout_qubits"):
        GlobalFloquetPartialShadowReservoir(config)


def test_manual_partition_filters_memory_readout_couplings():
    from pyqres.prethermal_shadow import GlobalFloquetConfig
    from pyqres.prethermal_shadow.dynamics import generate_hamiltonian_parameters

    config = GlobalFloquetConfig(
        n_qubits=4,
        n_memory=2,
        n_readout=2,
        omega=14.0,
        n_cycles_per_step=1,
        topology="all_to_all",
        include_mr_couplings=False,
        readout_qubits=(1, 3),
    )
    assert generate_hamiltonian_parameters(config)["edges"] == ((0, 2), (1, 3))


def test_symbolic_pauli_terms_match_dense_operator():
    from pyqres.prethermal_shadow.dynamics import parse_pauli_operator, parse_pauli_terms, pauli_terms_matrix

    spec = "0.5*0:X,2:Z + -1.2*1:Y + 0.25*0:X,2:Z"
    expected = parse_pauli_operator(3, spec, normalize=True)
    actual = pauli_terms_matrix(3, parse_pauli_terms(spec, normalize=True))

    assert np.allclose(actual, expected)


def test_prethermal_circuit_builds_measurement_reset_stream():
    pytest.importorskip("qiskit")
    from pyqres.prethermal_shadow import (
        GlobalFloquetConfig,
        InputEncodingConfig,
        PartialShadowReadoutConfig,
        QiskitGlobalFloquetPartialShadowReservoir,
    )

    reservoir = QiskitGlobalFloquetPartialShadowReservoir(
        GlobalFloquetConfig(
            n_qubits=3,
            n_memory=2,
            n_readout=1,
            readout_qubits=(1,),
            omega=12.0,
            n_cycles_per_step=2,
            random_h=False,
            random_jz=False,
            random_jxy=False,
            random_drive=False,
        ),
        InputEncodingConfig(operator="0:X + 2:X", beta=0.1, random_beta=False),
        PartialShadowReadoutConfig(pauli_k=1, shots=4),
    )
    circuit = reservoir.build_streaming_circuit([0.1, -0.2], [["X"], ["Y"]])

    assert reservoir.memory_qubits == (0, 2)
    assert reservoir.readout_qubits == (1,)
    assert circuit.num_clbits == 2
    assert circuit.count_ops()["measure"] == 2
    assert circuit.count_ops()["reset"] == 2
    assert circuit.count_ops()["PauliEvolution"] == 10


def test_prethermal_circuit_step_matches_dense_for_commuting_terms():
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Operator
    from pyqres.prethermal_shadow import (
        GlobalFloquetConfig,
        GlobalFloquetPartialShadowReservoir,
        InputEncodingConfig,
        PartialShadowReadoutConfig,
        QiskitGlobalFloquetPartialShadowReservoir,
    )
    from pyqres.prethermal_shadow.dynamics import reorder_qubit_operator

    floquet = GlobalFloquetConfig(
        n_qubits=2,
        n_memory=1,
        n_readout=1,
        omega=9.0,
        n_cycles_per_step=2,
        drive_axis="z",
        jxy_scale=0.0,
        break_scale=0.0,
        random_h=False,
        random_jz=False,
        random_jxy=False,
        random_drive=False,
        random_break=False,
    )
    encoding = InputEncodingConfig(input_qubits="all", axis="z", beta=0.13, random_beta=False)
    shadow = PartialShadowReadoutConfig(pauli_k=1, shots=4)
    dense = GlobalFloquetPartialShadowReservoir(floquet, encoding, shadow)
    circuit = QiskitGlobalFloquetPartialShadowReservoir(floquet, encoding, shadow)

    qiskit_unitary = Operator(circuit.build_step_circuit(0.17).decompose(reps=10)).data
    actual = reorder_qubit_operator(qiskit_unitary, (1, 0), (0, 1))
    expected = dense.u_floquet_step @ dense._input_unitary(0.17)
    phase = np.vdot(expected.reshape(-1), actual.reshape(-1))
    actual = actual / (phase / abs(phase))

    assert np.allclose(actual, expected, atol=1e-10)


def test_prethermal_mps_execution_smoke():
    pytest.importorskip("qiskit_aer")
    from pyqres.prethermal_shadow import (
        GlobalFloquetConfig,
        InputEncodingConfig,
        PartialShadowReadoutConfig,
        PrethermalCircuitConfig,
        QiskitGlobalFloquetPartialShadowReservoir,
    )

    reservoir = QiskitGlobalFloquetPartialShadowReservoir(
        GlobalFloquetConfig(
            n_qubits=3,
            n_memory=2,
            n_readout=1,
            omega=12.0,
            n_cycles_per_step=1,
            random_h=False,
            random_jz=False,
            random_jxy=False,
            random_drive=False,
        ),
        InputEncodingConfig(beta=0.05, random_beta=False),
        PartialShadowReadoutConfig(pauli_k=1, shots=6, seed=71, return_raw_shots=True),
        circuit_config=PrethermalCircuitConfig(
            simulator_method="matrix_product_state",
            circuit_batch_size=3,
            shots_per_basis=2,
            seed_simulator=72,
        ),
    )
    progress_updates = []
    features = reservoir.run_stream([0.1, -0.2], progress_callback=progress_updates.append)

    assert features.shape == (2, 4)
    assert np.all(np.isfinite(features))
    assert reservoir.last_basis_schedules.shape == (6, 2, 1)
    assert reservoir.last_outcomes.shape == (6, 2, 1)
    assert sum(progress_updates) == reservoir.basis_circuit_count() == 3


def test_standard_factory_builds_prethermal_mps_backend():
    pytest.importorskip("qiskit")
    import pyqres as qres

    reservoir = qres.qresreservoir.from_dict(
        {
            "preset": "prethermal_shadow",
            "memory_qubits": 2,
            "readout_qubits": 1,
            "backend": "qiskit",
            "floquet": {"omega": 12.0, "n_cycles_per_step": 1},
            "shadow": {"pauli_k": 1, "shots": 4},
            "qiskit": {"simulator_method": "matrix_product_state", "circuit_batch_size": 2},
        }
    )

    assert isinstance(reservoir, qres.QiskitGlobalFloquetPartialShadowReservoir)
    assert reservoir.circuit_config.simulator_method == "matrix_product_state"
