"""Small exact-feasible OVD scaling sweep for global-Floquet partial shadows."""

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


def feature_count_le2(n_readout: int) -> int:
    return 3 * int(n_readout) + 9 * int(n_readout) * (int(n_readout) - 1) // 2


def main() -> None:
    inputs = np.random.default_rng(0).normal(size=70)
    rows = []
    for n_qubits in [4, 5, 6, 7, 8]:
        n_readout = max(1, int(np.floor(0.33 * n_qubits)))
        n_memory = n_qubits - n_readout
        res = GlobalFloquetPartialShadowReservoir(
            GlobalFloquetConfig(n_qubits=n_qubits, n_memory=n_memory, n_readout=n_readout, omega=16.0, n_cycles_per_step=1, seed=23),
            InputEncodingConfig(beta=0.08, seed=29),
            PartialShadowReadoutConfig(pauli_k=min(2, n_readout), shots=256, exact_expectations=True, return_shadow_estimates=False, seed=31),
            ReadoutResetConfig(reset_state="zero"),
        )
        features = res.run(inputs)
        threshold = shadow_noise_threshold(features.shape[1], res.shadow_config.shots, res.shadow_config.pauli_k, 1.0)
        ovd = empirical_volterra_ovd(inputs, features, lag_horizon=3, max_order=2, tol=1e-8, finite_shadow_threshold=threshold)
        rows.append(
            {
                "n_qubits": n_qubits,
                "n_memory": n_memory,
                "n_readout": n_readout,
                "d_le2": feature_count_le2(n_readout),
                "feature_dim": int(features.shape[1]),
                "ideal_ovd": int(ovd["ovd_rank"]),
                "finite_shadow_ovd": int(ovd["finite_shadow_ovd"]),
            }
        )
    out = Path("outputs/ovd_scaling.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
