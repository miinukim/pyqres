"""Sweep omega/J for global-Floquet partial-shadow diagnostics."""

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
from pyqres.prethermal_shadow.diagnostics import spectrum_summary


def make_reservoir(omega: float) -> GlobalFloquetPartialShadowReservoir:
    return GlobalFloquetPartialShadowReservoir(
        GlobalFloquetConfig(n_qubits=5, n_memory=3, n_readout=2, omega=omega, n_cycles_per_step=2, seed=23),
        InputEncodingConfig(beta=0.08, seed=29),
        PartialShadowReadoutConfig(pauli_k=2, shots=128, exact_expectations=True, return_shadow_estimates=False, seed=31),
        ReadoutResetConfig(reset_state="zero"),
        seed_simulator=37,
    )


def main() -> None:
    out = Path("outputs/sweep_omega_prethermal.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for omega in [4.0, 8.0, 12.0, 16.0, 24.0, 32.0]:
        res = make_reservoir(omega)
        branch_q = res.branch_sensitivity(0.0, 0.05, horizon=6, observable_scope="full", pauli_k=1)
        branch_r = res.branch_sensitivity(0.0, 0.05, horizon=6, observable_scope="readout", pauli_k=1)
        spec = spectrum_summary(res.memory_channel_spectrum(pauli_k=1), res.delta_t)
        rows.append(
            {
                "omega": omega,
                "branch_full_final": float(branch_q[-1]),
                "branch_readout_final": float(branch_r[-1]),
                "max_lambda_abs": float(np.max(spec["lambda_abs"])),
                "min_decay_rate": float(np.min(spec["decay_rate"])),
            }
        )
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
