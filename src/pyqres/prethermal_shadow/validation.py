"""Mechanism-level diagnostics for the persistent-memory Floquet design.

The definitions follow Eqs. (5)-(8), (13), (16), and (19)-(24) of the
persistent-memory prethermal QRC construction.  These routines intentionally
measure dynamics, channel persistence, and visibility without invoking a
downstream learning task.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np
import scipy.linalg as la

from .config import (
    GlobalFloquetConfig,
    InputEncodingConfig,
    PartialShadowReadoutConfig,
    ReadoutResetConfig,
)
from .dimension import PrethermalShadowDimensionModel
from .dynamics import (
    PAULI,
    build_fast_period_unitary,
    fast_period,
    generate_hamiltonian_parameters,
    kron_all,
    pauli_string,
    reorder_qubit_operator,
    resolve_qubit_partition,
)
from .reservoir import GlobalFloquetPartialShadowReservoir


@dataclass(frozen=True)
class AsymmetricAnchorMetadata:
    """Resolved indices and interface strengths for Eq. (19)'s design."""

    anchor_memory_index: int
    anchor_qubit: int
    readout_local_index: int
    readout_qubit: int
    interface_edge_index: int
    interface_jz: float
    interface_jxy: float


@dataclass(frozen=True)
class AnchorMode:
    """Adjoint-channel eigenmode selected by normalized anchor overlap."""

    eigenvalue: complex
    operator: np.ndarray
    anchor_operator: np.ndarray
    anchor_overlap: float
    dressing_norm: float
    residual: float
    spectral_isolation: float
    subleading_radius: float
    eigenbasis_condition: float
    coefficients: np.ndarray
    left_coefficients: np.ndarray

    @property
    def channel_gap(self) -> float:
        return float(max(0.0, 1.0 - abs(self.eigenvalue)))


def asymmetric_anchor_parameters(
    config: GlobalFloquetConfig,
    *,
    interface_jxy: float,
    interface_jz: float = 1.0,
    anchor_memory_index: int = -1,
    readout_local_index: int = 0,
    anchor_jxy: float | None = None,
    anchor_break: float | None = None,
    anchor_drive: float = 0.0,
    readout_drive: float | None = None,
    isolate_anchor_diagonal: bool = False,
    interface_readout_bulk_degree: int | None = None,
) -> tuple[dict[str, object], AsymmetricAnchorMetadata]:
    """Apply the asymmetric memory-readout interface of Eqs. (19)-(20).

    The anchor is a logical memory index and ``readout_local_index`` is an
    index within the ordered readout subsystem.  All noncommuting exchange
    terms touching the anchor are set to ``anchor_jxy`` (defaulting to the
    interface value), and the anchor drive and breaking field are suppressed.
    """

    weak = float(interface_jxy)
    if weak < 0.0 or not np.isfinite(weak):
        raise ValueError("interface_jxy must be finite and non-negative")
    if not np.isfinite(interface_jz):
        raise ValueError("interface_jz must be finite")

    memory_qubits, readout_qubits = resolve_qubit_partition(config)
    anchor_local = int(anchor_memory_index)
    if anchor_local < 0:
        anchor_local += len(memory_qubits)
    if anchor_local < 0 or anchor_local >= len(memory_qubits):
        raise ValueError("anchor_memory_index is outside the memory subsystem")
    readout_local = int(readout_local_index)
    if readout_local < 0:
        readout_local += len(readout_qubits)
    if readout_local < 0 or readout_local >= len(readout_qubits):
        raise ValueError("readout_local_index is outside the readout subsystem")

    anchor_qubit = int(memory_qubits[anchor_local])
    readout_qubit = int(readout_qubits[readout_local])
    if interface_readout_bulk_degree is not None and int(interface_readout_bulk_degree) < 0:
        raise ValueError("interface_readout_bulk_degree must be non-negative")
    retained_readout_neighbors = {
        qubit for qubit in readout_qubits if qubit != readout_qubit
    }
    if interface_readout_bulk_degree is not None:
        retained_readout_neighbors = set(
            sorted(retained_readout_neighbors)[: int(interface_readout_bulk_degree)]
        )
    parameters = generate_hamiltonian_parameters(config)
    edges = tuple(parameters["edges"])  # type: ignore[arg-type]
    interface = tuple(sorted((anchor_qubit, readout_qubit)))
    try:
        interface_edge_index = [tuple(sorted(edge)) for edge in edges].index(interface)
    except ValueError as exc:
        raise ValueError("the selected anchor and readout qubits are not coupled") from exc

    jz = np.asarray(parameters["jz"], dtype=float).copy()
    jxy = np.asarray(parameters["jxy"], dtype=float).copy()
    drive = np.asarray(parameters["drive_coeffs"], dtype=float).copy()
    breaking = np.asarray(parameters["break_coeffs"], dtype=float).copy()

    weak_anchor = weak if anchor_jxy is None else float(anchor_jxy)
    weak_break = weak if anchor_break is None else float(anchor_break)
    for edge_index, edge in enumerate(edges):
        if anchor_qubit in edge:
            jxy[edge_index] = weak_anchor
            if isolate_anchor_diagonal and edge_index != interface_edge_index:
                jz[edge_index] = 0.0
        if (
            interface_readout_bulk_degree is not None
            and readout_qubit in edge
            and edge_index != interface_edge_index
        ):
            other_qubit = edge[1] if edge[0] == readout_qubit else edge[0]
            if other_qubit not in retained_readout_neighbors:
                jz[edge_index] = 0.0
                jxy[edge_index] = 0.0
    jz[interface_edge_index] = float(interface_jz)
    jxy[interface_edge_index] = weak
    drive[anchor_qubit] = float(anchor_drive)
    drive[readout_qubit] = (
        float(config.drive_amplitude) if readout_drive is None else float(readout_drive)
    )
    breaking[anchor_qubit] = weak_break

    result: dict[str, object] = {
        "edges": edges,
        "h": np.asarray(parameters["h"], dtype=float).copy(),
        "jz": jz,
        "jxy": jxy,
        "drive_coeffs": drive,
        "break_coeffs": breaking,
    }
    metadata = AsymmetricAnchorMetadata(
        anchor_memory_index=anchor_local,
        anchor_qubit=anchor_qubit,
        readout_local_index=readout_local,
        readout_qubit=readout_qubit,
        interface_edge_index=interface_edge_index,
        interface_jz=float(interface_jz),
        interface_jxy=weak,
    )
    return result, metadata


def build_asymmetric_anchor_reservoir(
    *,
    n_memory: int,
    n_readout: int,
    omega: float,
    n_cycles_per_step: int,
    interface_jxy: float,
    interface_jz: float = 1.0,
    h_scale: float = 1.0,
    bulk_jz: float = 0.3,
    bulk_jxy: float = 0.15,
    drive_amplitude: float = 0.7,
    bulk_break: float = 0.01,
    anchor_memory_index: int = -1,
    readout_local_index: int = 0,
    other_anchor_jxy: float = 0.0,
    anchor_break: float = 0.0,
    anchor_drive: float = 0.0,
    readout_drive: float | None = None,
    topology: str = "chain",
    randomize_bulk: bool = False,
    isolate_anchor_diagonal: bool = False,
    interface_readout_bulk_degree: int | None = None,
    input_beta: float = -1.0,
    seed: int = 11,
) -> tuple[GlobalFloquetPartialShadowReservoir, AsymmetricAnchorMetadata]:
    """Build a dense exact reservoir implementing the asymmetric anchor design."""

    n_memory = int(n_memory)
    n_readout = int(n_readout)
    config = GlobalFloquetConfig(
        n_qubits=n_memory + n_readout,
        n_memory=n_memory,
        n_readout=n_readout,
        omega=float(omega),
        n_cycles_per_step=int(n_cycles_per_step),
        drive_amplitude=float(drive_amplitude),
        h_scale=float(h_scale),
        jz_scale=float(bulk_jz),
        jxy_scale=float(bulk_jxy),
        break_scale=float(bulk_break),
        topology=str(topology),
        random_h=bool(randomize_bulk),
        random_jz=bool(randomize_bulk),
        random_jxy=bool(randomize_bulk),
        random_drive=bool(randomize_bulk),
        random_drive_sign=bool(randomize_bulk),
        random_break=bool(randomize_bulk),
        seed=int(seed),
    )
    parameters, metadata = asymmetric_anchor_parameters(
        config,
        interface_jxy=float(interface_jxy),
        interface_jz=float(interface_jz),
        anchor_memory_index=int(anchor_memory_index),
        readout_local_index=int(readout_local_index),
        anchor_jxy=float(other_anchor_jxy),
        anchor_break=float(anchor_break),
        anchor_drive=float(anchor_drive),
        readout_drive=readout_drive,
        isolate_anchor_diagonal=bool(isolate_anchor_diagonal),
        interface_readout_bulk_degree=interface_readout_bulk_degree,
    )
    reservoir = GlobalFloquetPartialShadowReservoir(
        config,
        InputEncodingConfig(
            operator=f"{metadata.anchor_qubit}:Y",
            beta=float(input_beta),
            random_beta=False,
            seed=int(seed),
        ),
        PartialShadowReadoutConfig(
            pauli_k=min(2, n_readout),
            feature_scope="readout",
            include_bias=True,
            exact_expectations=True,
            return_shadow_estimates=False,
            seed=int(seed),
        ),
        ReadoutResetConfig(reset_state="zero"),
        hamiltonian_parameters=parameters,
    )
    return reservoir, metadata


def first_order_effective_hamiltonian(
    h0: np.ndarray, drive: np.ndarray, period: float
) -> np.ndarray:
    """Return Eq. (7)'s first-order BCH Hamiltonian for Eq. (5)."""

    static = np.asarray(h0, dtype=complex)
    driven = np.asarray(drive, dtype=complex)
    if static.shape != driven.shape or static.ndim != 2 or static.shape[0] != static.shape[1]:
        raise ValueError("h0 and drive must be square matrices with matching shapes")
    commutator = static @ driven - driven @ static
    effective = static - 0.25j * float(period) * commutator
    return 0.5 * (effective + effective.conj().T)


def phase_aligned_unitary_error(exact: np.ndarray, approximate: np.ndarray) -> dict[str, float]:
    """Return global-phase-insensitive Frobenius and spectral unitary errors."""

    exact_array = np.asarray(exact, dtype=complex)
    approximate_array = np.asarray(approximate, dtype=complex)
    if exact_array.shape != approximate_array.shape:
        raise ValueError("unitaries must have matching shapes")
    phase_overlap = np.vdot(approximate_array, exact_array)
    phase = 1.0 + 0.0j if abs(phase_overlap) == 0.0 else phase_overlap / abs(phase_overlap)
    difference = exact_array - phase * approximate_array
    return {
        "frobenius": float(la.norm(difference, ord="fro") / la.norm(exact_array, ord="fro")),
        "spectral": float(la.norm(difference, ord=2)),
    }


def heisenberg_evolve(unitary: np.ndarray, observable: np.ndarray) -> np.ndarray:
    """Return ``U dagger O U``."""

    evolution = np.asarray(unitary, dtype=complex)
    operator = np.asarray(observable, dtype=complex)
    return evolution.conj().T @ operator @ evolution


def normalized_operator_error(exact: np.ndarray, approximate: np.ndarray) -> float:
    """Return relative Hilbert-Schmidt error between two operators."""

    exact_array = np.asarray(exact, dtype=complex)
    approximate_array = np.asarray(approximate, dtype=complex)
    denominator = la.norm(exact_array, ord="fro")
    if denominator == 0.0:
        raise ValueError("exact operator has zero Hilbert-Schmidt norm")
    return float(la.norm(exact_array - approximate_array, ord="fro") / denominator)


def pauli_weight_distribution(operator: np.ndarray, n_qubits: int) -> np.ndarray:
    """Return normalized Hilbert-Schmidt content at every Pauli weight."""

    n_qubits = int(n_qubits)
    dim = 2**n_qubits
    value = np.asarray(operator, dtype=complex)
    if value.shape != (dim, dim):
        raise ValueError(f"operator must have shape {(dim, dim)}")
    distribution = np.zeros(n_qubits + 1, dtype=float)
    for labels in product(("I", "X", "Y", "Z"), repeat=n_qubits):
        basis = kron_all([PAULI[label] for label in labels])
        coefficient = np.vdot(basis, value) / dim
        distribution[sum(label != "I" for label in labels)] += float(abs(coefficient) ** 2)
    total = float(distribution.sum())
    if total > 0.0:
        distribution /= total
    return distribution


def floquet_effective_diagnostics(
    reservoir: GlobalFloquetPartialShadowReservoir,
    observables: Mapping[str, Sequence[tuple[int, str]]],
    *,
    periods: int = 1,
) -> dict[str, object]:
    """Compare exact square-drive and first-order effective evolution."""

    periods = int(periods)
    if periods < 1:
        raise ValueError("periods must be positive")
    period = fast_period(reservoir.floquet_config)
    exact_period = build_fast_period_unitary(
        reservoir.h0, reservoir.drive_hamiltonian, period
    )
    effective_hamiltonian = first_order_effective_hamiltonian(
        reservoir.h0, reservoir.drive_hamiltonian, period
    )
    exact_unitary = np.linalg.matrix_power(exact_period, periods)
    effective_unitary = la.expm(-1j * periods * period * effective_hamiltonian)
    observable_results: dict[str, dict[str, object]] = {}
    for name, placements in observables.items():
        operator = pauli_string(reservoir.n_qubits, placements)
        exact_evolved = heisenberg_evolve(exact_unitary, operator)
        effective_evolved = heisenberg_evolve(effective_unitary, operator)
        observable_results[str(name)] = {
            "relative_error": normalized_operator_error(exact_evolved, effective_evolved),
            "exact_weight_distribution": pauli_weight_distribution(
                exact_evolved, reservoir.n_qubits
            ),
            "effective_weight_distribution": pauli_weight_distribution(
                effective_evolved, reservoir.n_qubits
            ),
        }
    return {
        "period": float(period),
        "periods": periods,
        "unitary_error": phase_aligned_unitary_error(exact_unitary, effective_unitary),
        "observables": observable_results,
    }


def normalized_hilbert_schmidt_overlap(left: np.ndarray, right: np.ndarray) -> float:
    """Return the magnitude of the normalized Hilbert-Schmidt overlap."""

    left_array = np.asarray(left, dtype=complex)
    right_array = np.asarray(right, dtype=complex)
    denominator = la.norm(left_array, ord="fro") * la.norm(right_array, ord="fro")
    if denominator == 0.0:
        return 0.0
    return float(abs(np.vdot(left_array, right_array)) / denominator)


def reset_covariance_matrix(
    reset_state: np.ndarray,
    readout_operators: Sequence[np.ndarray],
) -> np.ndarray:
    """Return the reset covariance matrix in Eqs. (S36)-(S37).

    The ordering convention is ``C[alpha, beta] = Tr(rho_R dB_alpha
    dB_beta)``. For a zero-state reset and ``(X, Y, Z)``, this gives the
    paper's matrix in Eq. (S53), with nonzero eigenvalue two.
    """

    reset = np.asarray(reset_state, dtype=complex)
    if reset.ndim != 2 or reset.shape[0] != reset.shape[1]:
        raise ValueError("reset_state must be a square matrix")
    if not np.allclose(reset, reset.conj().T):
        raise ValueError("reset_state must be Hermitian")
    if not np.isclose(np.trace(reset), 1.0):
        raise ValueError("reset_state must have unit trace")
    if np.min(la.eigvalsh(reset)) < -1e-12:
        raise ValueError("reset_state must be positive semidefinite")

    identity = np.eye(reset.shape[0], dtype=complex)
    centered: list[np.ndarray] = []
    for operator in readout_operators:
        value = np.asarray(operator, dtype=complex)
        if value.shape != reset.shape:
            raise ValueError("every readout operator must match reset_state")
        mean = np.trace(reset @ value)
        centered.append(value - mean * identity)
    return np.asarray(
        [
            [np.trace(reset @ left @ right) for right in centered]
            for left in centered
        ],
        dtype=complex,
    )


def _column_superoperator_from_kraus(kraus: Sequence[np.ndarray]) -> np.ndarray:
    """Return a channel matrix acting on column-major vectorized operators."""

    operators = [np.asarray(operator, dtype=complex) for operator in kraus]
    if not operators:
        raise ValueError("at least one Kraus operator is required")
    shape = operators[0].shape
    if shape[0] != shape[1] or any(operator.shape != shape for operator in operators):
        raise ValueError("Kraus operators must be equally sized square matrices")
    return sum(
        (np.kron(operator.conj(), operator) for operator in operators),
        start=np.zeros((shape[0] ** 2, shape[0] ** 2), dtype=complex),
    )


def _gksl_liouvillian(
    hamiltonian: np.ndarray,
    jump_operators: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return coherent and dissipative column-vectorized generators."""

    coherent_hamiltonian = np.asarray(hamiltonian, dtype=complex)
    dim = coherent_hamiltonian.shape[0]
    if coherent_hamiltonian.shape != (dim, dim):
        raise ValueError("hamiltonian must be square")
    identity = np.eye(dim, dtype=complex)
    coherent = -1j * (
        np.kron(identity, coherent_hamiltonian)
        - np.kron(coherent_hamiltonian.T, identity)
    )
    dissipative = np.zeros_like(coherent)
    for jump in jump_operators:
        operator = np.asarray(jump, dtype=complex)
        if operator.shape != (dim, dim):
            raise ValueError("jump operators must match hamiltonian")
        rate = operator.conj().T @ operator
        dissipative += np.kron(operator.conj(), operator)
        dissipative -= 0.5 * np.kron(identity, rate)
        dissipative -= 0.5 * np.kron(rate.T, identity)
    return coherent, dissipative


def _superoperator_choi(superoperator: np.ndarray, dim: int) -> np.ndarray:
    """Convert a column-vectorized superoperator into its Choi matrix."""

    matrix = np.asarray(superoperator, dtype=complex)
    if matrix.shape != (dim**2, dim**2):
        raise ValueError("superoperator has incompatible shape")
    choi = np.zeros((dim**2, dim**2), dtype=complex)
    for row in range(dim):
        for column in range(dim):
            basis = np.zeros((dim, dim), dtype=complex)
            basis[row, column] = 1.0
            mapped = (matrix @ basis.reshape(-1, order="F")).reshape(
                dim, dim, order="F"
            )
            choi[
                row * dim : (row + 1) * dim,
                column * dim : (column + 1) * dim,
            ] = mapped
    return 0.5 * (choi + choi.conj().T)


def paper_interface_gksl_diagnostics(
    collision_times: Sequence[float],
    *,
    interface_jxy: float,
    directed_eta: float,
    interface_jz: float = 0.0,
    reverse_eta: float = 0.0,
    memory_field: float = 0.0,
    readout_field: float = 0.0,
) -> dict[str, object]:
    """Validate the finite-collision GKSL construction in Eqs. (S38)-(S56).

    This is the minimal one-memory/one-readout interface from Eq. (S30), in
    memory-first tensor order. The exact comparison channel is generated by
    ``exp(-i * delta_t * H)`` and then traces a readout qubit initialized in
    ``|0><0|``. The paper-faithful approximation is

    ``exp(delta_t * L_H + delta_t**2 * D_reset)``.

    Its expansion is Eq. (S43), including the coherent second-order term. It
    is a finite-step collision approximation: the dissipative contribution per
    collision is quadratic in ``delta_t`` and is not treated as a fixed-rate
    continuous-time semigroup.
    """

    times = np.asarray(tuple(float(value) for value in collision_times), dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("collision_times must contain at least two values")
    if np.any(~np.isfinite(times)) or np.any(times <= 0.0):
        raise ValueError("collision_times must be finite and positive")

    x, y, z = (np.asarray(PAULI[axis], dtype=complex) for axis in ("X", "Y", "Z"))
    identity = np.asarray(PAULI["I"], dtype=complex)
    reset = np.zeros((2, 2), dtype=complex)
    reset[0, 0] = 1.0
    covariance = reset_covariance_matrix(reset, (x, y, z))

    g = float(interface_jxy)
    eta = float(directed_eta)
    crossing = (
        g * (np.kron(x, x) + np.kron(y, y))
        + eta * np.kron(z, y)
        + float(interface_jz) * np.kron(z, z)
        + float(reverse_eta) * np.kron(y, z)
    )
    full_hamiltonian = (
        float(memory_field) * np.kron(z, identity)
        + float(readout_field) * np.kron(identity, z)
        + crossing
    )
    full_hamiltonian = 0.5 * (full_hamiltonian + full_hamiltonian.conj().T)
    blocks = full_hamiltonian.reshape(2, 2, 2, 2)
    coherent_hamiltonian = blocks[:, 0, :, 0]
    block_jumps = tuple(blocks[:, output, :, 0] for output in range(1, 2))
    analytic_jump = g * (x + 1j * y) + 1j * eta * z
    coherent_generator, dissipative_generator = _gksl_liouvillian(
        coherent_hamiltonian, block_jumps
    )

    identity_superoperator = np.eye(4, dtype=complex)
    identity_vector = identity.reshape(-1, order="F")
    records: list[dict[str, float]] = []
    for collision_time in times:
        unitary = la.expm(-1j * collision_time * full_hamiltonian)
        unitary_blocks = unitary.reshape(2, 2, 2, 2)
        exact_kraus = tuple(unitary_blocks[:, output, :, 0] for output in range(2))
        exact_superoperator = _column_superoperator_from_kraus(exact_kraus)

        gksl_superoperator = la.expm(
            collision_time * coherent_generator
            + collision_time**2 * dissipative_generator
        )
        second_order_superoperator = (
            identity_superoperator
            + collision_time * coherent_generator
            + collision_time**2
            * (
                0.5 * coherent_generator @ coherent_generator
                + dissipative_generator
            )
        )
        main_text_shorthand = (
            identity_superoperator
            + collision_time * coherent_generator
            + collision_time**2 * dissipative_generator
        )

        exact_norm = max(float(la.norm(exact_superoperator, ord="fro")), 1e-15)
        gksl_error = float(
            la.norm(exact_superoperator - gksl_superoperator, ord="fro") / exact_norm
        )
        second_order_error = float(
            la.norm(exact_superoperator - second_order_superoperator, ord="fro")
            / exact_norm
        )
        main_text_shorthand_error = float(
            la.norm(exact_superoperator - main_text_shorthand, ord="fro")
            / exact_norm
        )
        gksl_choi = _superoperator_choi(gksl_superoperator, 2)
        exact_choi = _superoperator_choi(exact_superoperator, 2)
        records.append(
            {
                "collision_time": float(collision_time),
                "exact_vs_gksl_error": gksl_error,
                "exact_vs_second_order_error": second_order_error,
                "exact_vs_literal_main_text_error": main_text_shorthand_error,
                "gksl_error_over_dt_cubed": float(gksl_error / collision_time**3),
                "second_order_error_over_dt_cubed": float(
                    second_order_error / collision_time**3
                ),
                "exact_tp_residual": float(
                    la.norm(
                        exact_superoperator.conj().T @ identity_vector - identity_vector
                    )
                ),
                "gksl_tp_residual": float(
                    la.norm(
                        gksl_superoperator.conj().T @ identity_vector - identity_vector
                    )
                ),
                "exact_choi_min_eigenvalue": float(np.min(la.eigvalsh(exact_choi))),
                "gksl_choi_min_eigenvalue": float(np.min(la.eigvalsh(gksl_choi))),
            }
        )

    log_times = np.log(times)
    gksl_errors = np.asarray(
        [record["exact_vs_gksl_error"] for record in records], dtype=float
    )
    second_order_errors = np.asarray(
        [record["exact_vs_second_order_error"] for record in records], dtype=float
    )
    main_text_errors = np.asarray(
        [record["exact_vs_literal_main_text_error"] for record in records], dtype=float
    )
    gksl_exponent = float(np.polyfit(log_times, np.log(gksl_errors), 1)[0])
    second_order_exponent = float(
        np.polyfit(log_times, np.log(second_order_errors), 1)[0]
    )
    main_text_exponent = float(np.polyfit(log_times, np.log(main_text_errors), 1)[0])
    jump_denominator = max(float(la.norm(analytic_jump, ord="fro")), 1e-15)
    return {
        "covariance_matrix": covariance,
        "covariance_eigenvalues": la.eigvalsh(covariance),
        "coherent_hamiltonian": coherent_hamiltonian,
        "block_jump_operator": block_jumps[0],
        "analytic_jump_operator": analytic_jump,
        "jump_formula_relative_error": float(
            la.norm(block_jumps[0] - analytic_jump, ord="fro") / jump_denominator
        ),
        "gksl_error_scaling_exponent": gksl_exponent,
        "second_order_error_scaling_exponent": second_order_exponent,
        "literal_main_text_error_scaling_exponent": main_text_exponent,
        "records": records,
    }


def reservoir_step_gksl_diagnostics(
    reservoir: GlobalFloquetPartialShadowReservoir,
    *,
    duration_fractions: Sequence[float] = (
        1.0 / 128.0,
        1.0 / 64.0,
        1.0 / 32.0,
        1.0 / 16.0,
        1.0 / 8.0,
        1.0 / 4.0,
        1.0 / 2.0,
        1.0,
    ),
    cycle_counts: Sequence[int] | None = None,
) -> dict[str, object]:
    """Test the GKSL expansion using an STM reservoir's actual parameters.

    The first-order Floquet Hamiltonian is constructed from the reservoir's
    ``H0``, drive, and fast period. Its exact reduced collision channel is
    compared with the finite-collision GKSL completion. Separately, integer
    powers of the exact square-drive period are compared with both channels.
    This separates Floquet truncation error from collision-expansion error.

    The calculation is at zero input, where the STM input kick is the identity.
    The readout is reset only after the entire configured cycle block, so the
    practical collision duration is ``n_cycles_per_step * 2*pi/omega``.
    """

    if str(reservoir.reset_config.reset_state) != "zero":
        raise ValueError("the paper GKSL test currently requires reset_state='zero'")
    if not np.isclose(float(reservoir.input_config.bias), 0.0):
        raise ValueError("the reference-input test requires zero input bias")

    fractions = np.asarray(
        tuple(float(value) for value in duration_fractions), dtype=float
    )
    if fractions.ndim != 1 or fractions.size < 2:
        raise ValueError("duration_fractions must contain at least two values")
    if np.any(~np.isfinite(fractions)) or np.any(fractions <= 0.0):
        raise ValueError("duration_fractions must be finite and positive")
    fractions = np.unique(np.sort(fractions))

    configured_cycles = int(reservoir.floquet_config.n_cycles_per_step)
    cycles = (
        tuple(range(1, configured_cycles + 1))
        if cycle_counts is None
        else tuple(int(value) for value in cycle_counts)
    )
    if not cycles or any(value < 1 for value in cycles):
        raise ValueError("cycle_counts must contain positive integers")

    period = fast_period(reservoir.floquet_config)
    practical_duration = configured_cycles * period
    effective_physical = first_order_effective_hamiltonian(
        reservoir.h0, reservoir.drive_hamiltonian, period
    )
    effective = reorder_qubit_operator(
        effective_physical,
        tuple(range(reservoir.n_qubits)),
        reservoir.subsystem_qubit_order,
    )
    dim_memory = int(reservoir.dim_memory)
    dim_readout = int(reservoir.dim_readout)
    blocks = effective.reshape(
        dim_memory, dim_readout, dim_memory, dim_readout
    )
    coherent_hamiltonian = blocks[:, 0, :, 0]
    jump_operators = tuple(
        blocks[:, output, :, 0] for output in range(1, dim_readout)
    )
    coherent_generator, dissipative_generator = _gksl_liouvillian(
        coherent_hamiltonian, jump_operators
    )
    identity_memory = np.eye(dim_memory, dtype=complex)
    identity_vector = identity_memory.reshape(-1, order="F")

    def collision_maps(duration: float) -> tuple[np.ndarray, np.ndarray]:
        effective_unitary = la.expm(-1j * duration * effective)
        unitary_blocks = effective_unitary.reshape(
            dim_memory, dim_readout, dim_memory, dim_readout
        )
        effective_kraus = tuple(
            unitary_blocks[:, output, :, 0] for output in range(dim_readout)
        )
        exact_effective = _column_superoperator_from_kraus(effective_kraus)
        gksl = la.expm(
            duration * coherent_generator + duration**2 * dissipative_generator
        )
        return exact_effective, gksl

    def relative_error(left: np.ndarray, right: np.ndarray) -> float:
        denominator = max(float(la.norm(left, ord="fro")), 1e-15)
        return float(la.norm(left - right, ord="fro") / denominator)

    fraction_records: list[dict[str, float]] = []
    for fraction in fractions:
        duration = float(fraction * practical_duration)
        exact_effective, gksl = collision_maps(duration)
        choi = _superoperator_choi(gksl, dim_memory)
        error = relative_error(exact_effective, gksl)
        fraction_records.append(
            {
                "duration_fraction": float(fraction),
                "collision_time": duration,
                "effective_collision_vs_gksl_error": error,
                "error_over_dt_cubed": float(error / duration**3),
                "gksl_tp_residual": float(
                    la.norm(gksl.conj().T @ identity_vector - identity_vector)
                ),
                "gksl_choi_min_eigenvalue": float(np.min(la.eigvalsh(choi))),
            }
        )

    exact_period_physical = build_fast_period_unitary(
        reservoir.h0, reservoir.drive_hamiltonian, period
    )
    exact_period = reorder_qubit_operator(
        exact_period_physical,
        tuple(range(reservoir.n_qubits)),
        reservoir.subsystem_qubit_order,
    )
    cycle_records: list[dict[str, float | int]] = []
    for count in cycles:
        duration = count * period
        exact_unitary = np.linalg.matrix_power(exact_period, count)
        exact_blocks = exact_unitary.reshape(
            dim_memory, dim_readout, dim_memory, dim_readout
        )
        exact_kraus = tuple(
            exact_blocks[:, output, :, 0] for output in range(dim_readout)
        )
        exact_floquet = _column_superoperator_from_kraus(exact_kraus)
        exact_effective, gksl = collision_maps(duration)
        cycle_records.append(
            {
                "n_cycles": int(count),
                "collision_time": float(duration),
                "exact_floquet_vs_effective_channel_error": relative_error(
                    exact_floquet, exact_effective
                ),
                "effective_collision_vs_gksl_error": relative_error(
                    exact_effective, gksl
                ),
                "exact_floquet_vs_gksl_error": relative_error(exact_floquet, gksl),
            }
        )

    fit_count = min(5, len(fraction_records))
    fit_times = np.asarray(
        [record["collision_time"] for record in fraction_records[:fit_count]],
        dtype=float,
    )
    fit_errors = np.asarray(
        [
            record["effective_collision_vs_gksl_error"]
            for record in fraction_records[:fit_count]
        ],
        dtype=float,
    )
    return {
        "floquet_period": float(period),
        "configured_cycles_per_step": configured_cycles,
        "practical_collision_duration": float(practical_duration),
        "small_duration_error_exponent": float(
            np.polyfit(np.log(fit_times), np.log(fit_errors), 1)[0]
        ),
        "fraction_records": fraction_records,
        "cycle_records": cycle_records,
    }


def identify_anchor_mode(
    model: PrethermalShadowDimensionModel,
    *,
    anchor_memory_index: int,
    reference_input: float = 0.0,
    reference_operator: np.ndarray | None = None,
) -> AnchorMode:
    """Track Eq. (13)'s slow mode by overlap with an anchor reference.

    With ``reference_operator=None`` the reference is ``Z_m``.  Passing the
    mode at a nearby coupling implements adiabatic branch tracking from the
    exactly conserved ``Jxy_mr=0`` limit.
    """

    anchor = int(anchor_memory_index)
    if anchor < 0:
        anchor += model.n_memory
    if anchor < 0 or anchor >= model.n_memory:
        raise ValueError("anchor_memory_index is outside the memory subsystem")
    anchor_operator = pauli_string(model.n_memory, [(anchor, "Z")])
    reference = (
        anchor_operator
        if reference_operator is None
        else np.asarray(reference_operator, dtype=complex)
    )
    if reference.shape != anchor_operator.shape:
        raise ValueError("reference_operator must act on the memory subsystem")
    # The reset channel is generally non-unital.  Its adjoint therefore maps a
    # traceless observable to a traceless observable plus an identity offset.
    # Constants carry no state-dependent memory signal, so Eq. (13)'s modes
    # must be identified in the quotient of observables modulo the identity.
    # In the Pauli representation this is exactly the traceless PTM block.
    adjoint_ptm = model.ptm(float(reference_input)).conj().T
    traceless_adjoint = adjoint_ptm[1:, 1:]
    eigenvalues, eigenvectors = la.eig(traceless_adjoint)
    basis = model.memory_basis[1:]

    best_index = -1
    best_overlap = -1.0
    best_operator: np.ndarray | None = None
    for index in range(eigenvectors.shape[1]):
        coefficients = eigenvectors[:, index]
        operator = sum(
            (coefficient * basis_element for coefficient, basis_element in zip(coefficients, basis)),
            start=np.zeros_like(anchor_operator),
        )
        overlap = normalized_hilbert_schmidt_overlap(operator, reference)
        if overlap > best_overlap:
            best_index = index
            best_overlap = overlap
            best_operator = operator
    if best_operator is None:
        raise RuntimeError("failed to identify an adjoint-channel eigenoperator")

    operator_norm = la.norm(best_operator, ord="fro")
    best_operator = best_operator / operator_norm
    phase_anchor = np.vdot(best_operator, anchor_operator)
    if abs(phase_anchor) > 0.0:
        best_operator = best_operator * np.exp(1j * np.angle(phase_anchor))
    eigenvalue = complex(eigenvalues[best_index])
    residual_operator = (
        model.channel_adjoint(float(reference_input), best_operator)
        - eigenvalue * best_operator
    )
    residual_operator -= np.trace(residual_operator) / model.dim_memory * np.eye(
        model.dim_memory
    )
    residual = la.norm(residual_operator, ord="fro")
    overlap = normalized_hilbert_schmidt_overlap(best_operator, anchor_operator)
    distances = np.abs(eigenvalues - eigenvalue)
    distances[best_index] = np.inf
    spectral_isolation = float(np.min(distances)) if distances.size > 1 else np.inf
    remaining_radii = np.abs(eigenvalues).copy()
    remaining_radii[best_index] = -np.inf
    subleading_radius = float(np.max(remaining_radii)) if remaining_radii.size > 1 else 0.0
    coefficient_basis = eigenvectors
    eigenbasis_condition = float(np.linalg.cond(coefficient_basis))
    left_coefficients = np.sqrt(model.dim_memory) * np.linalg.inv(coefficient_basis)[
        best_index
    ]
    return AnchorMode(
        eigenvalue=eigenvalue,
        operator=best_operator,
        anchor_operator=anchor_operator,
        anchor_overlap=overlap,
        dressing_norm=float(np.sqrt(max(0.0, 1.0 - overlap**2))),
        residual=float(residual),
        spectral_isolation=spectral_isolation,
        subleading_radius=subleading_radius,
        eigenbasis_condition=eigenbasis_condition,
        coefficients=np.asarray(eigenvectors[:, best_index], dtype=complex),
        left_coefficients=np.asarray(left_coefficients, dtype=complex),
    )


def slow_mode_visibility(
    model: PrethermalShadowDimensionModel,
    mode: AnchorMode,
    *,
    readout_local_index: int,
    readout_axis: str = "X",
    reference_input: float = 0.0,
) -> dict[str, object]:
    """Evaluate Eq. (16)'s backward image and its overlap with ``Q_m``."""

    readout_index = int(readout_local_index)
    if readout_index < 0:
        readout_index += model.n_readout
    if readout_index < 0 or readout_index >= model.n_readout:
        raise ValueError("readout_local_index is outside the readout subsystem")
    readout_operator = pauli_string(
        model.n_readout, [(readout_index, str(readout_axis).upper())]
    )
    backward = model.backward_image_readout(readout_operator, float(reference_input))
    raw_projection = np.vdot(mode.operator, backward) / np.vdot(
        mode.operator, mode.operator
    )
    coefficients = np.asarray(
        [np.trace(basis.conj().T @ backward) / model.dim_memory for basis in model.memory_basis[1:]],
        dtype=complex,
    )
    biorthogonal_projection = np.dot(mode.left_coefficients, coefficients)
    return {
        "backward_image": backward,
        "normalized_visibility": normalized_hilbert_schmidt_overlap(
            mode.operator, backward
        ),
        "projection_abs": float(abs(raw_projection)),
        "biorthogonal_projection_abs": float(abs(biorthogonal_projection)),
        "backward_image_norm": float(la.norm(backward, ord="fro")),
    }


def impulse_response_diagnostics(
    reservoir: GlobalFloquetPartialShadowReservoir,
    metadata: AsymmetricAnchorMetadata,
    *,
    input_amplitude: float,
    max_delay: int | None = None,
    reference_input: float = 0.0,
    mode: AnchorMode | None = None,
    tail_fraction: float = 0.2,
    delays: Sequence[int] | None = None,
) -> dict[str, object]:
    """Measure the exact delayed local response after one anchor input pulse.

    The memory starts at the exact reference-input fixed point.  A one-step
    input pulse is applied, followed by reference-input updates.  ``X_r`` is
    evaluated through Eq. (16), so this is an exact channel-level impulse
    response rather than a trained-task score.  The modal curve is calibrated
    at delay one and propagated with the exact selected eigenvalue.
    """

    if delays is None:
        if max_delay is None or int(max_delay) < 1:
            raise ValueError("max_delay must be positive when delays is omitted")
        delay_values = np.arange(1, int(max_delay) + 1, dtype=int)
    else:
        delay_values = np.asarray(tuple(int(delay) for delay in delays), dtype=int)
        if delay_values.ndim != 1 or delay_values.size == 0 or np.any(delay_values < 1):
            raise ValueError("delays must be a non-empty sequence of positive integers")
        if np.any(np.diff(delay_values) <= 0):
            raise ValueError("delays must be strictly increasing")
    if not 0.0 < float(tail_fraction) <= 1.0:
        raise ValueError("tail_fraction must lie in (0, 1]")
    model = PrethermalShadowDimensionModel(reservoir)
    selected_mode = mode or identify_anchor_mode(
        model,
        anchor_memory_index=metadata.anchor_memory_index,
        reference_input=float(reference_input),
    )
    visibility = slow_mode_visibility(
        model,
        selected_mode,
        readout_local_index=metadata.readout_local_index,
        reference_input=float(reference_input),
    )
    stationary = model.fixed_point()
    delta = model.channel(float(input_amplitude), stationary) - stationary
    source_amplitude = np.trace(selected_mode.operator.conj().T @ delta)
    state_delta_coefficients = np.asarray(
        [
            np.trace(basis.conj().T @ delta) / model.dim_memory
            for basis in model.memory_basis[1:]
        ],
        dtype=complex,
    )
    backward = np.asarray(visibility["backward_image"], dtype=complex)
    backward_coefficients = np.asarray(
        [
            np.trace(basis.conj().T @ backward) / model.dim_memory
            for basis in model.memory_basis[1:]
        ],
        dtype=complex,
    )
    forward_ptm = model.ptm(float(reference_input))[1:, 1:]
    state_eigenvalues, state_eigenvectors = la.eig(forward_ptm)
    state_index = int(
        np.argmin(np.abs(state_eigenvalues - selected_mode.eigenvalue.conjugate()))
    )
    state_mode = state_eigenvectors[:, state_index]
    overlap = np.vdot(selected_mode.coefficients, state_mode)
    if abs(overlap) < 1e-12:
        raise RuntimeError("selected observable mode has no resolvable state dual")
    state_amplitude = np.vdot(selected_mode.coefficients, state_delta_coefficients) / overlap
    readout_amplitude = model.dim_memory * np.vdot(backward_coefficients, state_mode)
    exact = np.asarray(
        [
            model.dim_memory
            * np.vdot(
                backward_coefficients,
                np.linalg.matrix_power(forward_ptm, int(delay) - 1)
                @ state_delta_coefficients,
            )
            for delay in delay_values
        ],
        dtype=complex,
    )
    modal = readout_amplitude * state_amplitude * np.power(
        selected_mode.eigenvalue, delay_values - 1
    )
    denominator = max(float(la.norm(exact)), 1e-15)
    tail_start = max(0, int((1.0 - float(tail_fraction)) * len(delay_values)))
    tail_denominator = max(float(la.norm(exact[tail_start:])), 1e-15)
    terminal_denominator = max(float(abs(exact[-1])), 1e-15)
    return {
        "delays": delay_values,
        "exact_signal": exact,
        "modal_signal": modal,
        "relative_modal_error": float(la.norm(exact - modal) / denominator),
        "tail_relative_modal_error": float(
            la.norm(exact[tail_start:] - modal[tail_start:]) / tail_denominator
        ),
        "terminal_relative_modal_error": float(
            abs(exact[-1] - modal[-1]) / terminal_denominator
        ),
        "source_amplitude_abs": float(abs(source_amplitude)),
        "state_mode_amplitude_abs": float(abs(state_amplitude)),
        "readout_mode_amplitude_abs": float(abs(readout_amplitude)),
        "mode": selected_mode,
        "visibility": visibility,
    }


def slow_mode_diagnostics(
    reservoir: GlobalFloquetPartialShadowReservoir,
    metadata: AsymmetricAnchorMetadata,
    *,
    reference_input: float = 0.0,
    reference_operator: np.ndarray | None = None,
) -> dict[str, object]:
    """Return persistence, anchor dressing, and local readout visibility."""

    model = PrethermalShadowDimensionModel(reservoir)
    mode = identify_anchor_mode(
        model,
        anchor_memory_index=metadata.anchor_memory_index,
        reference_input=float(reference_input),
        reference_operator=reference_operator,
    )
    visibility = slow_mode_visibility(
        model,
        mode,
        readout_local_index=metadata.readout_local_index,
        reference_input=float(reference_input),
    )
    return {
        "eigenvalue": mode.eigenvalue,
        "lambda_abs": float(abs(mode.eigenvalue)),
        "channel_gap": mode.channel_gap,
        "anchor_overlap": mode.anchor_overlap,
        "dressing_norm": mode.dressing_norm,
        "eigen_residual": mode.residual,
        "spectral_isolation": mode.spectral_isolation,
        "subleading_radius": mode.subleading_radius,
        "eigenbasis_condition": mode.eigenbasis_condition,
        "normalized_visibility": visibility["normalized_visibility"],
        "visibility_projection_abs": visibility["projection_abs"],
        "visibility_biorthogonal_abs": visibility["biorthogonal_projection_abs"],
        "backward_image_norm": visibility["backward_image_norm"],
        "mode_operator": mode.operator,
        "backward_image": visibility["backward_image"],
    }


__all__ = [
    "AnchorMode",
    "AsymmetricAnchorMetadata",
    "asymmetric_anchor_parameters",
    "build_asymmetric_anchor_reservoir",
    "first_order_effective_hamiltonian",
    "floquet_effective_diagnostics",
    "heisenberg_evolve",
    "identify_anchor_mode",
    "impulse_response_diagnostics",
    "normalized_hilbert_schmidt_overlap",
    "normalized_operator_error",
    "paper_interface_gksl_diagnostics",
    "pauli_weight_distribution",
    "phase_aligned_unitary_error",
    "reservoir_step_gksl_diagnostics",
    "reset_covariance_matrix",
    "slow_mode_diagnostics",
    "slow_mode_visibility",
]
