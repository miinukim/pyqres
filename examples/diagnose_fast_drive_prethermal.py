"""Run fast-drive prethermal reservoir diagnostics and write artifacts.

The diagnostics are intentionally small by default so they can be used before
running a Mackey-Glass task benchmark:

    python examples/diagnose_fast_drive_prethermal.py
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from pyqres.experimental.prethermal_shadow import (
    FastDriveConfig,
    InputWriteConfig,
    PrethermalFloquetConfig,
    PrethermalShadowConfig,
    PrethermalShadowReservoir,
    ShadowReadoutConfig,
    TransducerConfig,
)
from pyqres.experimental.prethermal_shadow.diagnostics import (
    compare_shadow_to_exact,
    exact_readout_features,
    projected_memory_spectrum,
    run_memory_survival,
    run_transducer_sensitivity,
)


def default_config() -> PrethermalShadowConfig:
    return PrethermalShadowConfig(
        n_memory=3,
        n_readout=2,
        floquet=PrethermalFloquetConfig(
            fast_drive=FastDriveConfig(
                omega=16.0,
                n_cycles_per_input=2,
                drive_amplitude=0.8,
                drive_axis="x",
                drive_pattern="random",
                drive_seed=41,
            ),
            seed=23,
        ),
        input_write=InputWriteConfig(axis="y", beta=0.08),
        transducer=TransducerConfig(tau_c=0.06, seed=29),
        shadow=ShadowReadoutConfig(pauli_k=2, shots=24, include_bias=True, seed=31),
        simulator_method="density_matrix",
        seed_simulator=37,
    )


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(path: Path, rows: list[dict[str, float]], x_key: str, y_key: str, group_key: str | None = None) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    if group_key is None:
        ax.plot([row[x_key] for row in rows], [row[y_key] for row in rows], marker="o")
    else:
        groups = sorted({row[group_key] for row in rows})
        for group in groups:
            subset = [row for row in rows if row[group_key] == group]
            ax.plot([row[x_key] for row in subset], [row[y_key] for row in subset], marker="o", label=f"{group_key}={group:g}")
        ax.legend()
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def shadow_reference_rows(cfg: PrethermalShadowConfig) -> list[dict[str, float]]:
    """Compare finite-shot shadows to exact pre-measurement readout features."""

    inputs = np.linspace(-0.2, 0.2, 5)
    finite = PrethermalShadowReservoir(cfg).run_stream(inputs)
    reference = exact_readout_features(cfg, inputs)
    labels = PrethermalShadowReservoir(cfg).feature_labels
    rows = []
    for col, label in enumerate(labels):
        rows.append(
            {
                "feature_index": float(col),
                "finite_mean": float(np.mean(finite[:, col])),
                "exact_mean": float(np.mean(reference[:, col])),
                "mean_abs_error": float(np.mean(np.abs(finite[:, col] - reference[:, col]))),
            }
        )
    return rows


def shadow_estimator_sanity_rows(cfg: PrethermalShadowConfig) -> list[dict[str, float]]:
    """Pure reconstruction sanity check with synthetic exact features."""

    rng = np.random.default_rng(123)
    schedules = rng.choice(np.asarray(["X", "Y", "Z"], dtype="U1"), size=(64, 4, cfg.n_readout))
    outcomes = rng.integers(0, 2, size=schedules.shape, dtype=np.int8)
    exact = rng.normal(size=(4, 9))
    stats = compare_shadow_to_exact(exact, schedules, outcomes, n_readout=cfg.n_readout, pauli_k=cfg.shadow.pauli_k)
    return [
        {
            "feature_index": 0.0,
            "finite_mean": stats["correlation"],
            "exact_mean": 0.0,
            "mean_abs_error": stats["shadow_error_norm"],
        }
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/fast_drive_prethermal")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = default_config()
    (out / "fast_drive_config.json").write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True), encoding="utf-8")

    omega_values = [8.0, 16.0, 32.0]
    tau_values = [0.02, 0.06, 0.12, 0.2]
    survival = run_memory_survival(cfg, omega_values, lags=8)
    spectrum = projected_memory_spectrum(cfg, omega_values)
    sensitivity = run_transducer_sensitivity(cfg, tau_values)
    try:
        shadow = shadow_reference_rows(cfg)
    except ImportError:
        shadow = shadow_estimator_sanity_rows(cfg)

    write_csv(out / "fast_drive_memory_survival_vs_omega.csv", survival)
    write_csv(out / "projected_memory_spectrum_vs_omega.csv", spectrum)
    write_csv(out / "transducer_sensitivity_after_fast_drive.csv", sensitivity)
    write_csv(out / "shadow_vs_exact_after_fast_drive.csv", shadow)

    if not args.no_plots:
        maybe_plot(out / "fast_drive_memory_survival_vs_omega.png", survival, "lag", "feature_diff_norm", "omega")
        maybe_plot(out / "projected_memory_spectrum_vs_omega.png", spectrum, "omega", "abs")
        maybe_plot(out / "transducer_sensitivity_after_fast_drive.png", sensitivity, "tau_c", "readout_feature_diff_norm")
        maybe_plot(out / "shadow_vs_exact_after_fast_drive.png", shadow, "feature_index", "mean_abs_error")

    print(f"wrote diagnostics to {out}")


if __name__ == "__main__":
    main()
