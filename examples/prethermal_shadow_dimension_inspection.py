"""Empirical OVD and memory-spectrum inspection for partial-shadow dynamics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pyqres.prethermal_shadow import (
    GlobalFloquetConfig,
    GlobalFloquetPartialShadowReservoir,
    InputEncodingConfig,
    PartialShadowReadoutConfig,
    ReadoutResetConfig,
)
from pyqres.prethermal_shadow.diagnostics import empirical_volterra_ovd, shadow_noise_threshold, spectrum_summary


def default_reservoir(*, exact: bool = True) -> GlobalFloquetPartialShadowReservoir:
    return GlobalFloquetPartialShadowReservoir(
        GlobalFloquetConfig(n_qubits=5, n_memory=3, n_readout=2, omega=16.0, n_cycles_per_step=2, seed=23),
        InputEncodingConfig(input_qubits="memory", axis="y", beta=0.08, seed=29),
        PartialShadowReadoutConfig(
            pauli_k=2,
            shots=256,
            include_bias=True,
            exact_expectations=exact,
            return_shadow_estimates=not exact,
            seed=31,
        ),
        ReadoutResetConfig(reset_state="zero"),
        seed_simulator=37,
    )


def finite_float(value) -> float | None:
    val = float(np.real_if_close(value))
    return val if np.isfinite(val) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/prethermal_shadow_dimension_inspection.json")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    inputs = np.random.default_rng(0).normal(size=80)
    exact = default_reservoir(exact=True)
    features = exact.run(inputs)
    ovd = empirical_volterra_ovd(inputs, features, lag_horizon=4, max_order=2, tol=1e-8)
    lams = exact.memory_channel_spectrum(pauli_k=1)
    spec = spectrum_summary(lams, exact.delta_t)
    threshold = shadow_noise_threshold(
        len(exact.get_feature_names()),
        exact.shadow_config.shots,
        exact.shadow_config.pauli_k,
        exact.shadow_config.weak_strength,
    )
    report = {
        "feature_shape": list(features.shape),
        "feature_names": exact.get_feature_names(),
        "ovd_rank": int(ovd["ovd_rank"]),
        "singular_values": [finite_float(x) for x in ovd["singular_values"][:12]],
        "shadow_noise_threshold": float(threshold),
        "spectrum": {
            "lambda_abs": [finite_float(x) for x in spec["lambda_abs"]],
            "lambda_phase": [finite_float(x) for x in spec["lambda_phase"]],
            "decay_rate": [finite_float(x) for x in spec["decay_rate"]],
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not args.no_write:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
