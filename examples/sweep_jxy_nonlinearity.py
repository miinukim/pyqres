"""Sweep Jxy and estimate empirical OVD for global-Floquet partial shadows."""

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
from pyqres.prethermal_shadow.diagnostics import empirical_volterra_ovd


def main() -> None:
    inputs = np.random.default_rng(0).normal(size=80)
    rows = []
    for jxy in [0.0, 0.02, 0.05, 0.1, 0.2]:
        res = GlobalFloquetPartialShadowReservoir(
            GlobalFloquetConfig(n_qubits=5, n_memory=3, n_readout=2, omega=16.0, n_cycles_per_step=2, jxy_scale=jxy, seed=23),
            InputEncodingConfig(beta=0.08, seed=29),
            PartialShadowReadoutConfig(pauli_k=2, exact_expectations=True, return_shadow_estimates=False, seed=31),
            ReadoutResetConfig(reset_state="zero"),
        )
        features = res.run(inputs)
        ovd1 = empirical_volterra_ovd(inputs, features, lag_horizon=4, max_order=1, tol=1e-8)
        ovd2 = empirical_volterra_ovd(inputs, features, lag_horizon=4, max_order=2, tol=1e-8)
        branch = res.branch_sensitivity(0.0, 0.05, horizon=6, observable_scope="readout", pauli_k=1)
        rows.append(
            {
                "jxy_scale": jxy,
                "first_order_ovd": int(ovd1["ovd_rank"]),
                "second_order_ovd": int(ovd2["ovd_rank"]),
                "branch_readout_final": float(branch[-1]),
            }
        )
    out = Path("outputs/sweep_jxy_nonlinearity.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
