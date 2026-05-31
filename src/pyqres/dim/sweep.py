from __future__ import annotations

"""Sweep configuration and experiment helpers.

The package separates model definition from experiment orchestration. The model
classes know how to simulate one reservoir for one parameter set; this module
knows how to:

- turn a structured config mapping into a family of parameter sets
- instantiate a model for each sweep point
- run the Volterra analysis and collect tabular results
- write a small set of summary plots
"""

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import VolterraAnalyzer
from .linalg_utils import NumericalStabilityError
from .model import (
    FloquetIsingReservoirParameters,
    HaarRandomReservoirModel,
    HaarRandomReservoirParameters,
    IsingReservoirModel,
    IsingReservoirParameters,
    ThreeStepFloquetIsingReservoirModel,
    TwoStepFloquetIsingReservoirModel,
)


class SweepFamilyProtocol(Protocol):
    """Minimal protocol for anything that can produce a model along a 1D sweep."""

    name: str
    sweep_parameter: str

    def parameters(self, value: float) -> Any: ...

    def build_model(self, params: Any) -> Any: ...

    def parameter_label(self) -> str: ...


MODEL_REGISTRY = {
    "ising": (IsingReservoirParameters, IsingReservoirModel),
    "haar_random": (HaarRandomReservoirParameters, HaarRandomReservoirModel),
    "two_step_floquet": (FloquetIsingReservoirParameters, TwoStepFloquetIsingReservoirModel),
    "three_step_floquet": (FloquetIsingReservoirParameters, ThreeStepFloquetIsingReservoirModel),
}

DEFAULT_SWEEP_LABELS = {
    "integrability_break": "integrability-breaking coupling λ",
    "tau": "stroboscopic time τ",
    "readout_coupling": "memory-readout coupling κ",
    "input_strength": "input strength η",
}


@dataclass(frozen=True)
class SweepRule:
    """One rule describing how a sweep value modifies a single parameter field."""

    field: str
    mode: str = "replace"
    scale: float = 1.0
    offset: float = 0.0

    def apply(self, base_value: Any, sweep_value: float) -> float:
        # Translate the abstract sweep value into the concrete parameter value for one field.
        delta = self.offset + self.scale * sweep_value
        if self.mode == "replace":
            return delta
        if self.mode == "add":
            return float(base_value) + delta
        if self.mode == "multiply":
            return float(base_value) * delta
        raise ValueError(f"Unsupported sweep rule mode '{self.mode}' for field '{self.field}'")


@dataclass
class ConfigurableSweep:
    """Concrete sweep family assembled from config data.

    ``base_params`` defines one reference parameter set. ``sweep_rules`` then say
    which fields should vary when the abstract sweep coordinate changes.
    """

    name: str
    sweep_parameter: str
    parameter_cls: type
    model_cls: type
    base_params: Mapping[str, Any]
    sweep_rules: Sequence[SweepRule]
    sweep_label: str | None = None

    def parameter_label(self) -> str:
        if self.sweep_label:
            return self.sweep_label
        return DEFAULT_SWEEP_LABELS.get(self.sweep_parameter, self.sweep_parameter)

    def parameters(self, value: float) -> Any:
        # Build one concrete dataclass instance for the current sweep value.
        valid_fields = {field.name for field in fields(self.parameter_cls)}
        overrides = dict(self.base_params)
        for rule in self.sweep_rules:
            if rule.field not in valid_fields:
                raise ValueError(
                    f"Field '{rule.field}' is not valid for parameter class {self.parameter_cls.__name__}"
                )
            base_value = overrides.get(rule.field)
            if base_value is None:
                raise ValueError(
                    f"Sweep rule for field '{rule.field}' requires that field to be present in base_params"
                )
            # Apply each rule on top of the base parameter set to form one sweep point.
            overrides[rule.field] = rule.apply(base_value, float(value))
        return self.parameter_cls(**overrides)

    def build_model(self, params: Any) -> Any:
        return self.model_cls(params)


def _normalize_mapping(obj: Mapping[str, Any] | Any) -> dict[str, Any]:
    # Hydra/OmegaConf containers can arrive here; normalize them to plain string-key dicts.
    return {str(key): value for key, value in dict(obj).items()}


def _sweep_rules_from_config(sweep_cfg: Mapping[str, Any]) -> list[SweepRule]:
    # Convert the raw config list into typed rule objects so downstream code stays simple.
    rules_cfg = sweep_cfg.get("rules", [])
    return [
        SweepRule(
            field=str(rule["field"]),
            mode=str(rule.get("mode", "replace")),
            scale=float(rule.get("scale", 1.0)),
            offset=float(rule.get("offset", 0.0)),
        )
        for rule in rules_cfg
    ]


def build_sweep(sweep_cfg: Mapping[str, Any]) -> SweepFamilyProtocol:
    """Build a sweep family from the current config schema.

    Expected structure:

    - ``family``: model family name in ``MODEL_REGISTRY``
    - ``name``: optional human-readable label
    - ``base_params``: mapping used to instantiate the model parameter dataclass
    - ``sweep.parameter``: logical name shown in outputs/plots
    - ``sweep.rules``: list of field updates driven by the sweep value
    """

    cfg = _normalize_mapping(sweep_cfg)
    family = str(cfg.get("family", "ising")).lower()
    try:
        parameter_cls, model_cls = MODEL_REGISTRY[family]
    except KeyError as exc:
        raise ValueError(f"Unsupported Hamiltonian family '{family}'") from exc

    if "base_params" not in cfg:
        raise ValueError("Sweep config must define 'base_params'")

    base_params = _normalize_mapping(cfg["base_params"])
    sweep_section = _normalize_mapping(cfg.get("sweep", {}))
    if "parameter" not in sweep_section:
        raise ValueError("Sweep config must define 'sweep.parameter'")

    name = str(cfg.get("name", family))
    sweep_parameter = str(sweep_section["parameter"])
    return ConfigurableSweep(
        name=name,
        sweep_parameter=sweep_parameter,
        parameter_cls=parameter_cls,
        model_cls=model_cls,
        base_params=base_params,
        sweep_rules=_sweep_rules_from_config(sweep_section),
        sweep_label=None if sweep_section.get("label") is None else str(sweep_section["label"]),
    )


class SweepExperiment:
    """Run the same analysis across many sweep values and collect the results."""

    def __init__(
        self,
        sweep: SweepFamilyProtocol,
        sweep_values: Sequence[float],
        max_order: int = 2,
        lag_horizon: int = 2,
        observable_preset: str = "z",
        custom_observables: Sequence[str] | None = None,
        n_shots: int = 20000,
        delta: float = 0.05,
        noise_scale: float = 1.0,
    ):
        self.sweep = sweep
        self.sweep_values = list(sweep_values)
        self.max_order = max_order
        self.lag_horizon = lag_horizon
        self.observable_preset = observable_preset
        self.custom_observables = list(custom_observables) if custom_observables is not None else []
        self.n_shots = n_shots
        self.delta = delta
        self.noise_scale = noise_scale

    def run(self) -> pd.DataFrame:
        # Each row of the output dataframe corresponds to one sweep value and one
        # full analyze() call. The stored JSON columns preserve richer outputs
        # without forcing a second sidecar file format.
        rows = []
        for value in self.sweep_values:
            params = self.sweep.parameters(float(value))
            try:
                # Each sweep point builds a fresh model because the dense cached operators depend on params.
                model = self.sweep.build_model(params)
                observables = model.default_memory_observables(
                    preset=self.observable_preset,
                    custom_specs=self.custom_observables,
                )
                # The sweep layer stays intentionally thin: it chooses the model
                # and observable family, then delegates all structural work to
                # the analyzer so experiment scripts remain declarative.
                analyzer = VolterraAnalyzer(
                    model,
                    observables=observables,
                    max_order=self.max_order,
                    lag_horizon=self.lag_horizon,
                )
                result = analyzer.analyze(
                    n_shots=self.n_shots,
                    delta=self.delta,
                    noise_scale=self.noise_scale,
                )
            except NumericalStabilityError as exc:
                raise NumericalStabilityError(
                    f"Numerical instability during {self.sweep.name} sweep at {self.sweep.sweep_parameter}={float(value)}"
                ) from exc
            angles = np.array(result.principal_angles_deg, dtype=float)
            rows.append(
                {
                    "hamiltonian": self.sweep.name,
                    "sweep_parameter": self.sweep.sweep_parameter,
                    "sweep_value": float(value),
                    "latent_dim": int(result.latent_dim),
                    "vvr": int(result.vvr),
                    "ovd": int(result.ovd),
                    "noise_threshold": float(result.noise_threshold),
                    "n_features": int(len(result.monomials)),
                    "n_observables": int(len(observables)),
                    "singular_values": json.dumps([float(x) for x in np.real_if_close(result.singular_values)]),
                    "restricted_singular_values": json.dumps(
                        [float(x) for x in np.real_if_close(result.restricted_singular_values)]
                    ),
                    "principal_angles_deg": json.dumps([float(x) for x in angles]),
                    "parameters": json.dumps(asdict(params)),
                }
            )
        return pd.DataFrame(rows)

    def save(self, df: pd.DataFrame, outdir: str | Path, stem: str = "ising_volterra_sweep") -> Path:
        """Persist a completed sweep table and quick-look diagnostic plots."""

        # Persist both the raw table and a couple of quick-look diagnostic plots.
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        csv_path = outdir / f"{stem}.csv"
        df.to_csv(csv_path, index=False)

        # Save summary plots alongside the raw CSV so sweep runs stay self-contained.
        grouped = "hamiltonian" in df.columns and df["hamiltonian"].nunique() > 1
        line_groups = df.groupby("hamiltonian", sort=False) if grouped else [("run", df)]

        fig1 = plt.figure(figsize=(7, 4))
        for label, group in line_groups:
            plt.plot(group["sweep_value"], group["latent_dim"], marker="o", label=f"{label} latent dim")
            plt.plot(group["sweep_value"], group["vvr"], marker="s", linestyle="--", label=f"{label} VVR")
            plt.plot(group["sweep_value"], group["ovd"], marker="^", linestyle=":", label=f"{label} OVD")
        plt.xlabel(self.sweep.parameter_label())
        plt.ylabel("dimension / rank")
        plt.legend()
        plt.tight_layout()
        fig1.savefig(outdir / f"{stem}_dimensions.png", dpi=180)
        plt.close(fig1)

        fig2 = plt.figure(figsize=(7, 4))
        for label, group in (df.groupby("hamiltonian", sort=False) if grouped else [("run", df)]):
            angle_lists = [json.loads(raw) for raw in group["principal_angles_deg"]]
            max_angles = max((len(angles) for angles in angle_lists), default=0)
            sweep_values = group["sweep_value"].to_numpy(dtype=float)
            for idx in range(max_angles):
                yvals = [angles[idx] if idx < len(angles) else np.nan for angles in angle_lists]
                plt.plot(
                    sweep_values,
                    yvals,
                    marker="o",
                    linewidth=1.0,
                    markersize=3.0,
                    label=f"{label} angle {idx + 1}",
                )
        plt.xlabel(self.sweep.parameter_label())
        plt.ylabel("angle (deg)")
        plt.legend()
        plt.tight_layout()
        fig2.savefig(outdir / f"{stem}_angles.png", dpi=180)
        plt.close(fig2)

        return csv_path
