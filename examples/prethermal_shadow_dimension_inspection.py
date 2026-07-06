"""Dimension and visibility inspection for prethermal shadow dynamics.

This script analyzes the deterministic memory-channel part of the prethermal
shadow reservoir:

    input write -> prethermal Floquet block -> transducer -> readout reset

The finite-shot random classical-shadow measurement layer is not itself a PTM
channel, so the pyqres.dim analyzers are run on an exact-channel proxy with the
same proposed dynamics and readout reset modeled as |+><+|. The reported
visibility/isotropy metrics use memory-observable readout matrices.

Run from the repository root:

    python examples/prethermal_shadow_dimension_inspection.py

Optionally write a report elsewhere:

    python examples/prethermal_shadow_dimension_inspection.py --output outputs/prethermal_dim.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from pyqres.dim.analysis import DenseVolterraAnalyzer, PTMAffineExpansion, VolterraAnalyzer, _ambient_readout_matrix
from pyqres.dim.isotropy import compressed_visibility_diagnostics
from pyqres.dim.model import ReservoirBase
from pyqres.experimental.prethermal_shadow import (
    FastDriveConfig,
    InputWriteConfig,
    PrethermalFloquetConfig,
    PrethermalShadowConfig,
    ShadowReadoutConfig,
    TransducerConfig,
)
from pyqres.experimental.prethermal_shadow.circuits import append_input_write, append_prethermal_block, append_transducer
from pyqres.experimental.prethermal_shadow.config import validate_and_resolve_config


class PrethermalExactProxyModel(ReservoirBase):
    """Exact dense PTM proxy for the prethermal shadow memory dynamics."""

    def __init__(self, cfg: PrethermalShadowConfig):
        self.resolved = validate_and_resolve_config(cfg)
        self.params = self.resolved.base
        self._initialize_common(
            self.resolved.base.n_memory,
            self.resolved.base.n_readout,
            reset_to_zero_state=False,
        )
        self.reset_state = plus_density(self.n_readout)

    def _build_unitary(self, u: float) -> np.ndarray:
        qc = QuantumCircuit(self.n_total)
        memory = tuple(range(self.n_memory))
        readout = tuple(range(self.n_memory, self.n_total))
        append_input_write(qc, memory, float(u), self.resolved)
        append_prethermal_block(qc, memory, self.resolved)
        append_transducer(qc, memory, readout, self.resolved)
        return np.asarray(Operator(qc).data, dtype=complex)


def plus_density(n_qubits: int) -> np.ndarray:
    """Return |+...+><+...+|."""

    plus = np.ones((2, 1), dtype=complex) / np.sqrt(2.0)
    rho = plus @ plus.conj().T
    out = rho
    for _ in range(1, int(n_qubits)):
        out = np.kron(out, rho)
    return out


def default_config() -> PrethermalShadowConfig:
    """Small default setup that keeps dense PTM analysis tractable."""

    return PrethermalShadowConfig(
        n_memory=2,
        n_readout=1,
        floquet=PrethermalFloquetConfig(
            mode="fast_drive",
            fast_drive=FastDriveConfig(omega=16.0, n_cycles_per_input=2, drive_amplitude=0.8, drive_seed=41),
            seed=23,
        ),
        input_write=InputWriteConfig(axis="y", beta=0.08, bias=0.0),
        transducer=TransducerConfig(tau_c=0.04, seed=29),
        shadow=ShadowReadoutConfig(pauli_k=1, shots=256, include_bias=True, seed=31),
        simulator_method="density_matrix",
        seed_simulator=37,
    )


def summarize_volterra(result) -> dict[str, object]:
    return {
        "latent_dim": int(result.latent_dim),
        "vvr": int(result.vvr),
        "ovd": int(result.ovd),
        "n_monomials": int(len(result.monomials)),
        "noise_threshold": float(result.noise_threshold),
        "singular_values": real_list(result.singular_values),
        "restricted_singular_values": real_list(result.restricted_singular_values),
        "principal_angles_deg": real_list(result.principal_angles_deg),
    }


def summarize_isotropy(diag) -> dict[str, object]:
    return {
        "ambient_dim": int(diag.ambient_dim),
        "r_visible": int(diag.r_visible),
        "s_gamma": int(diag.s_gamma),
        "exact_null_dim": int(diag.exact_null_dim),
        "supported_dim": int(diag.supported_dim),
        "alpha_ambient": finite_float(diag.alpha_ambient),
        "alpha_gamma": finite_float(diag.alpha_gamma),
        "alpha_plus": finite_float(diag.alpha_plus),
        "delta_iso_gamma": finite_float(diag.delta_iso_gamma),
        "delta_iso_plus": finite_float(diag.delta_iso_plus),
        "theta_gamma_star_deg": finite_float(diag.theta_gamma_star_deg),
        "theta_plus_star_deg": finite_float(diag.theta_plus_star_deg),
        "supported_visibility_angle_deg": real_list(diag.supported_visibility_angle_deg),
    }


def real_list(values: np.ndarray, limit: int = 12) -> list[float]:
    arr = np.asarray(values)
    return [finite_float(x) for x in np.real_if_close(arr[:limit])]


def finite_float(value) -> float | None:
    val = float(np.real_if_close(value))
    return val if np.isfinite(val) else None


def run_dimension_inspection(
    cfg: PrethermalShadowConfig,
    *,
    readout_presets: tuple[str, ...] = ("z", "xyz", "rich"),
    max_order: int = 2,
    lag_horizon: int = 2,
    fd_step: float = 2e-3,
    algebraic_tol: float = 1e-8,
    n_shots: int = 256,
    delta: float = 0.05,
) -> dict[str, object]:
    """Run dense PTM, observable-side, and compressed isotropy diagnostics."""

    model = PrethermalExactProxyModel(cfg)
    expansion = PTMAffineExpansion(
        model,
        max_order=max_order,
        fd_step=fd_step,
        expansion_point=0.0,
    )
    report: dict[str, object] = {
        "notes": [
            "Exact-channel proxy for prethermal shadow dynamics: input write + Floquet block + transducer + readout reset in |+>.",
            "Finite-shot randomized classical-shadow measurement is not a PTM channel and is not included in channel derivatives.",
            "Visibility/isotropy uses memory-observable readout matrices.",
        ],
        "setup": setup_summary(model),
        "ptm_expansion": {
            "T0_shape": list(model.ptm(0.0).shape),
            "A0_shape": list(expansion.A0.shape),
            "A1_fro_norm": float(np.linalg.norm(expansion.Ak(1))),
            "A2_fro_norm": float(np.linalg.norm(expansion.Ak(2))),
            "b1_norm": float(np.linalg.norm(expansion.bk(1))),
            "b2_norm": float(np.linalg.norm(expansion.bk(2))),
            "fixed_point_trace": [
                float(np.real(np.trace(expansion.rho_bar))),
                float(np.imag(np.trace(expansion.rho_bar))),
            ],
        },
        "readout_suites": {},
    }

    suites = report["readout_suites"]
    assert isinstance(suites, dict)
    for preset in readout_presets:
        observables = model.default_memory_observables(preset=preset)
        dense = DenseVolterraAnalyzer(
            model,
            observables=observables,
            max_order=max_order,
            lag_horizon=lag_horizon,
            fd_step=fd_step,
            algebraic_tol=algebraic_tol,
            expansion_point=0.0,
        ).analyze(n_shots=n_shots, delta=delta)
        observable_side = VolterraAnalyzer(
            model,
            observables=observables,
            max_order=max_order,
            lag_horizon=lag_horizon,
            fd_step=fd_step,
            algebraic_tol=algebraic_tol,
            expansion_point=0.0,
        ).analyze(n_shots=n_shots, delta=delta)
        dense_iso = compressed_visibility_diagnostics(
            dense.latent_basis_matrix,
            model.readout_matrix(observables),
            tol=algebraic_tol,
        )
        observable_iso = compressed_visibility_diagnostics(
            observable_side.latent_basis_matrix,
            _ambient_readout_matrix(observables),
            tol=algebraic_tol,
        )
        suites[preset] = {
            "n_observables": len(observables),
            "dense_volterra": summarize_volterra(dense),
            "dense_isotropy": summarize_isotropy(dense_iso),
            "observable_side_volterra": summarize_volterra(observable_side),
            "observable_side_isotropy": summarize_isotropy(observable_iso),
        }
    return report


def setup_summary(model: PrethermalExactProxyModel) -> dict[str, object]:
    cfg = model.params
    return {
        "n_memory": model.n_memory,
        "n_readout": model.n_readout,
        "dim_memory": model.dim_memory,
        "floquet_mode": cfg.floquet.mode,
        "n_floquet": cfg.floquet.n_floquet,
        "floquet_tau": cfg.floquet.tau,
        "fast_drive_omega": cfg.floquet.fast_drive.omega,
        "fast_drive_cycles_per_input": cfg.floquet.fast_drive.n_cycles_per_input,
        "fast_drive_period": model.resolved.fast_drive_period,
        "reservoir_dt": model.resolved.reservoir_dt,
        "input_axis": cfg.input_write.axis,
        "input_beta": cfg.input_write.beta,
        "tau_c": cfg.transducer.tau_c if cfg.transducer else None,
        "resolved_h": model.resolved.h.tolist(),
        "resolved_jz": model.resolved.jz.tolist(),
        "resolved_jxy": model.resolved.jxy.tolist(),
        "resolved_x_break": model.resolved.x_break.tolist(),
        "resolved_g": None if model.resolved.g is None else model.resolved.g.tolist(),
    }


def print_table(report: dict[str, object]) -> None:
    print("PTM expansion:")
    ptm = report["ptm_expansion"]
    assert isinstance(ptm, dict)
    for key in ("T0_shape", "A0_shape", "A1_fro_norm", "A2_fro_norm", "b1_norm", "b2_norm"):
        print(f"  {key}: {ptm[key]}")
    print()
    print("Readout summary:")
    print("preset | dense latent/VVR/OVD | observable latent/VVR/OVD | dense exact-null | obs exact-null")
    suites = report["readout_suites"]
    assert isinstance(suites, dict)
    for preset, data in suites.items():
        dense = data["dense_volterra"]
        obs = data["observable_side_volterra"]
        dense_iso = data["dense_isotropy"]
        obs_iso = data["observable_side_isotropy"]
        print(
            f"{preset:>5} | "
            f"{dense['latent_dim']}/{dense['vvr']}/{dense['ovd']} | "
            f"{obs['latent_dim']}/{obs['vvr']}/{obs['ovd']} | "
            f"{dense_iso['exact_null_dim']}/{dense_iso['s_gamma']} | "
            f"{obs_iso['exact_null_dim']}/{obs_iso['s_gamma']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/prethermal_shadow_dimension_inspection.json")
    parser.add_argument("--no-write", action="store_true", help="Print only; do not write JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_dimension_inspection(default_config())
    print_table(report)
    if not args.no_write:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
