"""Sweep weak-shadow strength and compare finite-shot features to exact features."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from pyqres.prethermal_shadow import (
    GlobalFloquetConfig,
    GlobalFloquetPartialShadowReservoir,
    InputEncodingConfig,
    PartialShadowReadoutConfig,
    ReadoutResetConfig,
)
from pyqres.prethermal_shadow.diagnostics import empirical_volterra_ovd, shadow_noise_threshold


def make_reservoir(strength: float, *, exact: bool = False) -> GlobalFloquetPartialShadowReservoir:
    return GlobalFloquetPartialShadowReservoir(
        GlobalFloquetConfig(n_qubits=5, n_memory=3, n_readout=2, omega=16.0, n_cycles_per_step=2, seed=23),
        InputEncodingConfig(beta=0.08, seed=29),
        PartialShadowReadoutConfig(
            pauli_k=2,
            shots=256,
            measurement_type="projective" if strength == 1.0 else "weak",
            weak_strength=strength,
            exact_expectations=exact,
            return_shadow_estimates=not exact,
            seed=31,
        ),
        ReadoutResetConfig(reset_state="zero"),
        seed_simulator=37,
    )


def main() -> None:
    inputs = np.random.default_rng(0).normal(size=80)
    exact = make_reservoir(1.0, exact=True).run(inputs)
    rows = []
    for strength in [1.0, 0.7, 0.5, 0.3, 0.2, 0.1]:
        res = make_reservoir(strength)
        features = res.run(inputs)
        threshold = shadow_noise_threshold(features.shape[1], res.shadow_config.shots, res.shadow_config.pauli_k, strength)
        ovd = empirical_volterra_ovd(inputs, features, lag_horizon=4, max_order=2, tol=1e-8, finite_shadow_threshold=threshold)
        rows.append(
            {
                "weak_strength": strength,
                "mean_l2_error": float(np.mean(np.linalg.norm(features - exact, axis=1))),
                "ideal_ovd": int(ovd["ovd_rank"]),
                "finite_shadow_ovd": int(ovd["finite_shadow_ovd"]),
                "shadow_threshold": float(threshold),
            }
        )
    out = Path("outputs/sweep_shadow_strength.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
