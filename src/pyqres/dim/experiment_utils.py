from __future__ import annotations

"""Lightweight helpers for writing pyqres dimension-analysis experiments with less boilerplate.

The goal of this module is not to replace custom experiment scripts. It covers
the common case where an experiment:

1. reads a Hydra config with ``sweep`` and ``experiment`` sections
2. builds models through ``build_sweep(...)``
3. runs ``IsingVolterraAnalyzer`` on each sweep value
4. stores a standard row schema with the main pyqres diagnostics
5. saves a CSV, resolved config, and a small number of line plots

Scripts with multiple model families, task-side evaluation, or custom
post-processing can still build on these helpers while overriding the parts they
need.
"""

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
import json
from typing import Any, Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from .analysis import IsingVolterraAnalyzer, VolterraResult
from .linalg_utils import NumericalStabilityError
from .sweep import SweepFamilyProtocol, build_sweep

AnalysisExtraRowFn = Callable[[VolterraResult, Any, Any, Sequence[np.ndarray], float], Mapping[str, Any]]
FailureExtraRowFn = Callable[[Exception, float], Mapping[str, Any]]
CovarianceModelFn = Callable[[Any, Any, Sequence[np.ndarray], float], float | np.ndarray | None]

_PAULIS = ("X", "Y", "Z")


@dataclass(frozen=True)
class LineMetricSpec:
    """Description of one metric line in generated summary plots."""

    metric: str
    label: str
    marker: str = "o"
    linestyle: str = "-"
    linewidth: float = 1.5


def sweep_values_from_cfg(cfg: DictConfig) -> np.ndarray:
    """Build a simple linspace sweep from ``cfg.sweep.grid``."""

    grid = cfg.sweep.grid
    return np.linspace(float(grid.start), float(grid.stop), int(grid.num))


def _cfg_float(cfg: DictConfig, key: str, default: float) -> float:
    return float(cfg[key]) if key in cfg else float(default)


def _cfg_int(cfg: DictConfig, key: str, default: int) -> int:
    return int(cfg[key]) if key in cfg else int(default)


def _pauli_spec_from_word(word: Sequence[str]) -> str:
    return "*".join(f"{pauli}{site}" for site, pauli in enumerate(word) if pauli != "I")


def _random_pauli_observable_specs(
    model: Any,
    random_cfg: DictConfig,
    *,
    exclude_specs: set[str] | None = None,
) -> list[str]:
    """Sample unique Pauli observable specs from a locality-limited pool."""

    count = int(random_cfg.get("count", 0))
    if count < 0:
        raise ValueError(f"Random observable count must be non-negative, got {count}.")
    if count == 0:
        return []

    n_memory = int(model.n_memory)
    max_locality_raw = random_cfg.get("max_locality", None)
    max_locality = n_memory if max_locality_raw is None else int(max_locality_raw)
    if not (1 <= max_locality <= n_memory):
        raise ValueError(
            f"random.max_locality must lie in [1, n_memory={n_memory}] or be null, got {max_locality_raw}."
        )

    exclude_identity = bool(random_cfg.get("exclude_identity", True))
    if not exclude_identity:
        raise ValueError("random.exclude_identity=false is not supported because observable specs require Pauli support.")
    excluded = set() if exclude_specs is None else set(exclude_specs)
    pool: list[str] = []
    for word in product(("I", *_PAULIS), repeat=n_memory):
        # The pool is built from Pauli words and then converted to compact specs
        # such as `X0*Z2`. Identity-only words are not valid observables here.
        locality = sum(pauli != "I" for pauli in word)
        if locality == 0:
            continue
        if locality <= max_locality:
            spec = _pauli_spec_from_word(word)
            if spec not in excluded:
                pool.append(spec)

    if count > len(pool):
        raise ValueError(
            f"Requested {count} random observables, but the random Pauli pool has only {len(pool)} available entries "
            f"for n_memory={n_memory}, max_locality={max_locality} after excluding preset/custom observables."
        )

    rng = np.random.default_rng(int(random_cfg.get("seed", 0)))
    indices = rng.choice(len(pool), size=count, replace=False)
    return [pool[int(index)] for index in indices]


def _observable_specs_from_cfg(model: Any, observables_cfg: DictConfig) -> list[str]:
    """Combine preset, custom, and optional random observable specs."""

    base_specs = model.default_memory_observable_specs(
        preset=str(observables_cfg.preset),
        custom_specs=list(observables_cfg.custom),
    )
    random_cfg = observables_cfg.get("random", None)
    if random_cfg is None or not bool(random_cfg.get("enabled", False)):
        return base_specs

    base_set = set(base_specs)
    random_specs = _random_pauli_observable_specs(model, random_cfg, exclude_specs=base_set)
    return list(dict.fromkeys([*base_specs, *random_specs]))


def _standard_analysis_row(
    *,
    sweep: SweepFamilyProtocol,
    sweep_value: float,
    params: Any,
    observables: Sequence[np.ndarray],
    observable_specs: Sequence[str],
    result: VolterraResult,
) -> dict[str, Any]:
    """Flatten a successful `VolterraResult` into one CSV-friendly row."""

    angles = np.array(result.principal_angles_deg, dtype=float)
    return {
        "status": "ok",
        "error": "",
        "family": sweep.name,
        "sweep_parameter": sweep.sweep_parameter,
        "sweep_parameter_label": sweep.parameter_label(),
        "sweep_value": float(sweep_value),
        "latent_dim": int(result.latent_dim),
        "vvr": int(result.vvr),
        "ovd": int(result.ovd),
        "whitened_ovd": int(result.whitened_ovd),
        "soft_ovd": float(result.soft_ovd),
        "visible_effective_rank": float(result.visible_effective_rank),
        "whitened_effective_rank": float(result.whitened_effective_rank),
        "noise_threshold": float(result.noise_threshold),
        "n_observables": int(len(observables)),
        "n_features": int(len(result.monomials)),
        "min_angle_deg": float(np.min(angles)) if angles.size else np.nan,
        "mean_angle_deg": float(np.mean(angles)) if angles.size else np.nan,
        "max_angle_deg": float(np.max(angles)) if angles.size else np.nan,
        "singular_values": json.dumps([float(x) for x in np.real_if_close(result.singular_values)]),
        "restricted_singular_values": json.dumps(
            [float(x) for x in np.real_if_close(result.restricted_singular_values)]
        ),
        "whitened_singular_values": json.dumps(
            [float(x) for x in np.real_if_close(result.whitened_singular_values)]
        ),
        "principal_angles_deg": json.dumps([float(x) for x in angles]),
        "observable_specs": json.dumps(list(observable_specs)),
        "parameters": json.dumps(asdict(params)),
    }


def _standard_failure_row(
    *,
    sweep: SweepFamilyProtocol,
    sweep_value: float,
    exc: Exception,
) -> dict[str, Any]:
    """Create a CSV-friendly row for a failed sweep point."""

    return {
        "status": "failed",
        "error": str(exc),
        "family": sweep.name,
        "sweep_parameter": sweep.sweep_parameter,
        "sweep_parameter_label": sweep.parameter_label(),
        "sweep_value": float(sweep_value),
        "latent_dim": np.nan,
        "vvr": np.nan,
        "ovd": np.nan,
        "whitened_ovd": np.nan,
        "soft_ovd": np.nan,
        "visible_effective_rank": np.nan,
        "whitened_effective_rank": np.nan,
        "noise_threshold": np.nan,
        "n_observables": np.nan,
        "n_features": np.nan,
        "min_angle_deg": np.nan,
        "mean_angle_deg": np.nan,
        "max_angle_deg": np.nan,
        "singular_values": "[]",
        "restricted_singular_values": "[]",
        "whitened_singular_values": "[]",
        "principal_angles_deg": "[]",
        "observable_specs": "[]",
        "parameters": "",
    }


def run_standard_analysis_sweep(
    cfg: DictConfig,
    *,
    sweep_values: Sequence[float] | None = None,
    extra_row_fn: AnalysisExtraRowFn | None = None,
    failure_row_fn: FailureExtraRowFn | None = None,
    covariance_model_fn: CovarianceModelFn | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the standard pyqres Ising analysis across a 1D sweep.

    This collapses the repetitive experiment pattern

    ``build_sweep -> build model -> choose observables -> build analyzer -> analyze``

    into one helper call. Callers can still extend each successful or failed row
    by passing small callback functions instead of rewriting the entire loop.
    """

    sweep = build_sweep(OmegaConf.to_container(cfg.sweep, resolve=True))
    values = list(sweep_values) if sweep_values is not None else list(sweep_values_from_cfg(cfg))
    rows: list[dict[str, Any]] = []
    total = len(values)

    for idx, sweep_value in enumerate(values, start=1):
        try:
            if verbose:
                print(
                    f"[{idx}/{total}] {sweep.name}: {sweep.sweep_parameter}={float(sweep_value):.8g} starting",
                    flush=True,
            )
            params = sweep.parameters(float(sweep_value))
            # Build a fresh model for each value because model instances cache
            # dense unitaries/PTMs keyed by input values.
            model = sweep.build_model(params)
            observable_specs = _observable_specs_from_cfg(model, cfg.experiment.observables)
            observables = [model.parse_memory_observable(spec) for spec in observable_specs]
            if verbose:
                print(
                    f"[{idx}/{total}] built model with {len(observables)} observables; "
                    f"P={_cfg_int(cfg.experiment, 'max_order', 2)}, "
                    f"L={_cfg_int(cfg.experiment, 'lag_horizon', 2)}",
                    flush=True,
                )

            analyzer = IsingVolterraAnalyzer(
                model,
                observables=observables,
                max_order=_cfg_int(cfg.experiment, "max_order", 2),
                lag_horizon=_cfg_int(cfg.experiment, "lag_horizon", 2),
                fd_step=_cfg_float(cfg.experiment, "fd_step", 5e-3),
                algebraic_tol=_cfg_float(cfg.experiment, "algebraic_tol", 1e-9),
                expansion_point=_cfg_float(cfg.experiment, "expansion_point", 0.0),
            )

            covariance_model = (
                None
                if covariance_model_fn is None
                else covariance_model_fn(params, model, observables, float(sweep_value))
            )
            result = analyzer.analyze(
                n_shots=_cfg_int(cfg.experiment, "n_shots", 2000),
                delta=_cfg_float(cfg.experiment, "delta", 0.05),
                noise_scale=_cfg_float(cfg.experiment, "noise_scale", 1.0),
                covariance_model=covariance_model,
            )

            row = _standard_analysis_row(
                sweep=sweep,
                sweep_value=float(sweep_value),
                params=params,
                observables=observables,
                observable_specs=observable_specs,
                result=result,
            )
            if extra_row_fn is not None:
                row.update(dict(extra_row_fn(result, params, model, observables, float(sweep_value))))
            rows.append(row)
            if verbose:
                print(
                    f"[{idx}/{total}] done: latent_dim={result.latent_dim}, "
                    f"vvr={result.vvr}, ovd={result.ovd}, "
                    f"soft_ovd={result.soft_ovd:.6g}",
                    flush=True,
                )
        except (MemoryError, NumericalStabilityError, ValueError) as exc:
            # The experiment table records failed values rather than aborting the
            # whole sweep, which is useful for scans over unstable regimes.
            row = _standard_failure_row(sweep=sweep, sweep_value=float(sweep_value), exc=exc)
            if failure_row_fn is not None:
                row.update(dict(failure_row_fn(exc, float(sweep_value))))
            rows.append(row)
            if verbose:
                print(f"[{idx}/{total}] failed: {exc}", flush=True)

    return pd.DataFrame(rows)


def save_experiment_table(
    cfg: DictConfig,
    df: pd.DataFrame,
    *,
    csv_name: str,
) -> tuple[Path, Path]:
    """Write the primary CSV and resolved Hydra config for one experiment."""

    outdir = Path(cfg.paths.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, outdir / "resolved_config.yaml", resolve=True)
    csv_path = outdir / csv_name
    df.to_csv(csv_path, index=False)
    return outdir, csv_path


def save_line_metric_plot(
    df: pd.DataFrame,
    outdir: Path,
    *,
    xcol: str,
    metrics: Sequence[LineMetricSpec],
    filename: str,
    xlabel: str | None = None,
    ylabel: str = "metric value",
    title: str | None = None,
    figsize: tuple[float, float] = (8.0, 4.5),
) -> Path:
    """Save a simple line plot for selected dataframe metrics."""

    fig = plt.figure(figsize=figsize)
    for spec in metrics:
        plt.plot(
            df[xcol],
            df[spec.metric],
            marker=spec.marker,
            linestyle=spec.linestyle,
            linewidth=spec.linewidth,
            label=spec.label,
        )
    plt.xlabel(xlabel if xlabel is not None else xcol)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.legend()
    path = outdir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


__all__ = [
    "AnalysisExtraRowFn",
    "CovarianceModelFn",
    "FailureExtraRowFn",
    "LineMetricSpec",
    "run_standard_analysis_sweep",
    "save_experiment_table",
    "save_line_metric_plot",
    "sweep_values_from_cfg",
]
