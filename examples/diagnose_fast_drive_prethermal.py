"""Write small diagnostics for global-Floquet partial-shadow dynamics."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
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


def make_reservoir(omega: float = 16.0, weak_strength: float = 1.0) -> GlobalFloquetPartialShadowReservoir:
    return GlobalFloquetPartialShadowReservoir(
        GlobalFloquetConfig(n_qubits=5, n_memory=3, n_readout=2, omega=omega, n_cycles_per_step=2, seed=23),
        InputEncodingConfig(input_qubits="memory", axis="y", beta=0.08, seed=29),
        PartialShadowReadoutConfig(
            pauli_k=2,
            shots=128,
            measurement_type="projective" if weak_strength == 1.0 else "weak",
            weak_strength=weak_strength,
            include_bias=True,
            seed=31,
        ),
        ReadoutResetConfig(reset_state="zero"),
        seed_simulator=37,
    )


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/global_floquet_partial_shadow")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    reservoir = make_reservoir()
    (out / "config.json").write_text(
        json.dumps(
            {
                "floquet": asdict(reservoir.floquet_config),
                "input": asdict(reservoir.input_config),
                "shadow": asdict(reservoir.shadow_config),
                "reset": asdict(reservoir.reset_config),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    branch_rows = []
    for scope in ("full", "memory", "readout"):
        vals = reservoir.branch_sensitivity(0.0, 0.05, horizon=8, observable_scope=scope, pauli_k=1)
        branch_rows.extend({"scope": scope, "lag": float(i), "distinguishability": float(v)} for i, v in enumerate(vals))
    write_csv(out / "branch_sensitivity.csv", branch_rows)

    lams = reservoir.memory_channel_spectrum(pauli_k=1)
    spec = spectrum_summary(lams, reservoir.delta_t)
    spectrum_rows = [
        {
            "index": float(i),
            "lambda_abs": float(spec["lambda_abs"][i]),
            "lambda_phase": float(spec["lambda_phase"][i]),
            "decay_rate": float(spec["decay_rate"][i]),
        }
        for i in range(len(lams))
    ]
    write_csv(out / "memory_channel_spectrum.csv", spectrum_rows)

    inputs = np.linspace(-0.2, 0.2, 6)
    exact_res = make_reservoir(weak_strength=1.0)
    exact_res.shadow_config = PartialShadowReadoutConfig(pauli_k=2, exact_expectations=True, return_shadow_estimates=False)
    exact = exact_res.run(inputs)
    shadow = make_reservoir(weak_strength=0.5).run(inputs)
    error_rows = [
        {"time": float(t), "l2_error": float(np.linalg.norm(shadow[t] - exact[t]))}
        for t in range(inputs.size)
    ]
    write_csv(out / "shadow_vs_exact_error.csv", error_rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
