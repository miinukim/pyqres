from __future__ import annotations

"""Diagnostics for explicit fast-drive prethermal dynamics."""

from collections.abc import Sequence
from dataclasses import replace
from itertools import combinations, product

import numpy as np
import scipy.linalg as la

from .config import PrethermalShadowConfig, ResolvedPrethermalShadowConfig, validate_and_resolve_config
from .dynamics import build_memory_step_unitary, kron_all, pair_memory_pauli, single_memory_pauli
from .shadows import estimate_shadow_features, generate_pauli_labels, sample_basis_schedules


def low_weight_memory_paulis(n_memory: int, max_weight: int = 2) -> list[tuple[tuple[int, str], np.ndarray]]:
    """Return memory Pauli operators up to the requested weight."""

    out: list[tuple[tuple[int, str], np.ndarray]] = []
    for weight in range(1, int(max_weight) + 1):
        for sites in combinations(range(int(n_memory)), weight):
            for paulis in product(("X", "Y", "Z"), repeat=weight):
                label = tuple((int(site), str(pauli)) for site, pauli in zip(sites, paulis))
                if weight == 1:
                    op = single_memory_pauli(n_memory, sites[0], paulis[0])
                else:
                    op = pair_memory_pauli(n_memory, sites[0], paulis[0], sites[1], paulis[1])
                    for site, pauli in zip(sites[2:], paulis[2:]):
                        op = op @ single_memory_pauli(n_memory, site, pauli)
                out.append((label, op))
    return out


def memory_pauli_features(rho: np.ndarray, operators: Sequence[tuple[tuple[int, str], np.ndarray]]) -> np.ndarray:
    """Return expectation values of selected memory Pauli operators."""

    return np.asarray([float(np.real_if_close(np.trace(op @ rho))) for _, op in operators], dtype=float)


def initial_memory_state(n_memory: int, kind: str = "zero", eps: float = 0.2) -> np.ndarray:
    """Return a non-maximally-mixed diagnostic initial state."""

    kind = str(kind)
    dim = 2 ** int(n_memory)
    if kind == "maximally_mixed":
        raise ValueError("memory survival diagnostics must not use a maximally mixed initial state.")
    if kind == "zero":
        rho = np.zeros((dim, dim), dtype=complex)
        rho[0, 0] = 1.0
        return rho
    if kind == "polarized":
        rho = np.eye(dim, dtype=complex)
        for i in range(int(n_memory)):
            rho += float(eps) * single_memory_pauli(n_memory, i, "Z")
        rho /= np.trace(rho)
        return 0.5 * (rho + rho.conj().T)
    raise ValueError("initial state must be zero, polarized, or maximally_mixed.")


def input_write_unitary(cfg: ResolvedPrethermalShadowConfig, u: float) -> np.ndarray:
    """Dense memory-only input write unitary matching the circuit convention."""

    n_memory = int(cfg.base.n_memory)
    dim = 2**n_memory
    out = np.eye(dim, dtype=complex)
    angle_scale = float(u + cfg.base.input_write.bias)
    axis = str(cfg.base.input_write.axis).upper()
    for i, beta in enumerate(cfg.beta):
        generator = single_memory_pauli(n_memory, i, axis)
        local = la.expm(-1j * float(beta) * angle_scale * generator)
        out = local @ out
    return out


def total_pauli(n_total: int, placements: Sequence[tuple[int, str]]) -> np.ndarray:
    """Return a dense Pauli product on the full memory+readout Hilbert space."""

    ops = [np.eye(2, dtype=complex) for _ in range(int(n_total))]
    paulis = {
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }
    for site, pauli in placements:
        ops[int(site)] = paulis[str(pauli).upper()]
    return kron_all(ops)


def plus_density(n_qubits: int) -> np.ndarray:
    """Return |+...+><+...+|."""

    rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
    out = np.array([[1.0 + 0.0j]])
    for _ in range(int(n_qubits)):
        out = np.kron(out, rho)
    return out


def trace_out_readout(rho: np.ndarray, n_memory: int, n_readout: int) -> np.ndarray:
    """Trace the readout subsystem from memory-first tensor ordering."""

    dim_memory = 2 ** int(n_memory)
    dim_readout = 2 ** int(n_readout)
    return np.trace(
        np.asarray(rho, dtype=complex).reshape(dim_memory, dim_readout, dim_memory, dim_readout),
        axis1=1,
        axis2=3,
    )


def transducer_unitary(cfg: ResolvedPrethermalShadowConfig) -> np.ndarray:
    """Dense full-system transducer unitary."""

    n_memory = int(cfg.base.n_memory)
    n_readout = int(cfg.base.n_readout)
    n_total = n_memory + n_readout
    dim = 2**n_total
    if cfg.base.transducer is None or cfg.g is None:
        return np.eye(dim, dtype=complex)
    tau_c = float(cfg.base.transducer.tau_c)
    if tau_c == 0.0:
        return np.eye(dim, dtype=complex)
    htr = np.zeros((dim, dim), dtype=complex)
    for alpha in range(n_readout):
        for i in range(n_memory):
            htr += float(cfg.g[alpha, i]) * total_pauli(n_total, [(i, "Z"), (n_memory + alpha, "Z")])
    return la.expm(-1j * tau_c * htr)


def exact_readout_features(
    cfg: PrethermalShadowConfig,
    inputs: Sequence[float],
    *,
    include_bias: bool | None = None,
) -> np.ndarray:
    """Return exact pre-measurement readout Pauli features for a stream."""

    resolved = validate_and_resolve_config(cfg)
    n_memory = int(resolved.base.n_memory)
    n_readout = int(resolved.base.n_readout)
    dim_readout = 2**n_readout
    labels = generate_pauli_labels(n_readout, int(resolved.base.shadow.pauli_k))
    include = bool(resolved.base.shadow.include_bias if include_bias is None else include_bias)
    rho_memory = initial_memory_state(n_memory, kind="zero")
    plus = plus_density(n_readout)
    u_memory_step = build_memory_step_unitary(resolved)
    u_transducer = transducer_unitary(resolved)
    observables = [
        total_pauli(n_memory + n_readout, [(n_memory + qubit, pauli) for qubit, pauli in label])
        for label in labels
    ]
    rows = []
    for value in np.asarray(inputs, dtype=float).reshape(-1):
        rho = np.kron(rho_memory, plus)
        u_input = input_write_unitary(resolved, float(value))
        u_memory = np.kron(u_memory_step @ u_input, np.eye(dim_readout, dtype=complex))
        rho = u_memory @ rho @ u_memory.conj().T
        rho = u_transducer @ rho @ u_transducer.conj().T
        row = [float(np.real_if_close(np.trace(obs @ rho))) for obs in observables]
        rows.append(([1.0] if include else []) + row)
        rho_memory = trace_out_readout(rho, n_memory, n_readout)
    return np.asarray(rows, dtype=float)


def run_memory_survival(
    cfg: PrethermalShadowConfig,
    omega_values: Sequence[float],
    *,
    lags: int = 12,
    u0: float = 0.0,
    delta: float = 0.05,
    initial_state: str = "zero",
    max_weight: int = 2,
) -> list[dict[str, float]]:
    """Measure low-weight memory input-survival versus fast-drive frequency."""

    rows: list[dict[str, float]] = []
    for omega in omega_values:
        local_cfg = replace(
            cfg,
            floquet=replace(
                cfg.floquet,
                mode="fast_drive",
                fast_drive=replace(cfg.floquet.fast_drive, omega=float(omega)),
            ),
        )
        resolved = validate_and_resolve_config(local_cfg)
        u_step = build_memory_step_unitary(resolved)
        rho0 = initial_memory_state(resolved.base.n_memory, kind=initial_state)
        u_plus = input_write_unitary(resolved, u0 + delta)
        u_minus = input_write_unitary(resolved, u0 - delta)
        rho_plus = u_plus @ rho0 @ u_plus.conj().T
        rho_minus = u_minus @ rho0 @ u_minus.conj().T
        ops = low_weight_memory_paulis(resolved.base.n_memory, max_weight=max_weight)
        for lag in range(int(lags) + 1):
            diff = memory_pauli_features(rho_plus, ops) - memory_pauli_features(rho_minus, ops)
            rows.append(
                {
                    "omega": float(omega),
                    "omega_over_j": float(omega) / max(float(np.max(np.abs(resolved.jz))), 1e-15),
                    "lag": float(lag),
                    "feature_diff_norm": float(np.linalg.norm(diff)),
                    "reservoir_dt": float(resolved.reservoir_dt),
                }
            )
            rho_plus = u_step @ rho_plus @ u_step.conj().T
            rho_minus = u_step @ rho_minus @ u_step.conj().T
    return rows


def projected_memory_spectrum(
    cfg: PrethermalShadowConfig,
    omega_values: Sequence[float],
    *,
    max_weight: int = 2,
) -> list[dict[str, float]]:
    """Projected Heisenberg spectrum on low-weight memory Paulis."""

    rows: list[dict[str, float]] = []
    for omega in omega_values:
        local_cfg = replace(
            cfg,
            floquet=replace(
                cfg.floquet,
                mode="fast_drive",
                fast_drive=replace(cfg.floquet.fast_drive, omega=float(omega)),
            ),
        )
        resolved = validate_and_resolve_config(local_cfg)
        u_step = build_memory_step_unitary(resolved)
        ops = low_weight_memory_paulis(resolved.base.n_memory, max_weight=max_weight)
        dim = 2 ** int(resolved.base.n_memory)
        tmat = np.zeros((len(ops), len(ops)), dtype=complex)
        for mu, (_, p_mu) in enumerate(ops):
            evolved = u_step.conj().T @ p_mu @ u_step
            for nu, (_, p_nu) in enumerate(ops):
                tmat[mu, nu] = np.trace(p_nu @ evolved) / dim
        eigvals = np.linalg.eigvals(tmat)
        for idx, eig in enumerate(eigvals):
            rows.append(
                {
                    "omega": float(omega),
                    "index": float(idx),
                    "real": float(np.real(eig)),
                    "imag": float(np.imag(eig)),
                    "abs": float(abs(eig)),
                    "theta": float(np.angle(eig)),
                }
            )
    return rows


def run_transducer_sensitivity(
    cfg: PrethermalShadowConfig,
    tau_c_values: Sequence[float],
    *,
    u0: float = 0.0,
    delta: float = 0.05,
    initial_state: str = "zero",
) -> list[dict[str, float]]:
    """Approximate exact readout sensitivity after verified memory survival."""

    rows: list[dict[str, float]] = []
    for tau_c in tau_c_values:
        local_cfg = replace(cfg, transducer=replace(cfg.transducer, tau_c=float(tau_c)) if cfg.transducer else None)
        resolved = validate_and_resolve_config(local_cfg)
        rho0 = initial_memory_state(resolved.base.n_memory, kind=initial_state)
        u_plus = build_memory_step_unitary(resolved) @ input_write_unitary(resolved, u0 + delta)
        u_minus = build_memory_step_unitary(resolved) @ input_write_unitary(resolved, u0 - delta)
        rho_plus = u_plus @ rho0 @ u_plus.conj().T
        rho_minus = u_minus @ rho0 @ u_minus.conj().T
        # First-order exact transducer signal proxy: coupled Z memories weighted by g.
        if resolved.g is None:
            signal = 0.0
        else:
            z_ops = [single_memory_pauli(resolved.base.n_memory, i, "Z") for i in range(resolved.base.n_memory)]
            plus_z = np.asarray([np.real_if_close(np.trace(op @ rho_plus)) for op in z_ops], dtype=float)
            minus_z = np.asarray([np.real_if_close(np.trace(op @ rho_minus)) for op in z_ops], dtype=float)
            readout_diff = 2.0 * float(tau_c) * (resolved.g @ (plus_z - minus_z))
            signal = float(np.linalg.norm(readout_diff))
        rows.append({"tau_c": float(tau_c), "readout_feature_diff_norm": signal})
    return rows


def compare_shadow_to_exact(
    exact_features: np.ndarray,
    shadow_schedules: np.ndarray,
    shadow_outcomes: np.ndarray,
    *,
    n_readout: int,
    pauli_k: int,
) -> dict[str, float]:
    """Compare exact readout features to reconstructed shadow features."""

    labels = generate_pauli_labels(n_readout, pauli_k)
    shadow_features = estimate_shadow_features(shadow_schedules, shadow_outcomes, labels, include_bias=False)
    exact = np.asarray(exact_features, dtype=float).reshape(-1)
    shadow = shadow_features.reshape(-1)
    n = min(exact.shape[0], shadow.shape[0])
    if n == 0:
        return {"correlation": float("nan"), "shadow_error_norm": float("nan")}
    corr = float(np.corrcoef(exact[:n], shadow[:n])[0, 1]) if n > 1 else float("nan")
    err = float(np.linalg.norm(exact[:n] - shadow[:n]))
    return {"correlation": corr, "shadow_error_norm": err}


__all__ = [
    "compare_shadow_to_exact",
    "exact_readout_features",
    "initial_memory_state",
    "input_write_unitary",
    "low_weight_memory_paulis",
    "memory_pauli_features",
    "projected_memory_spectrum",
    "run_memory_survival",
    "run_transducer_sensitivity",
    "total_pauli",
    "trace_out_readout",
    "transducer_unitary",
]
