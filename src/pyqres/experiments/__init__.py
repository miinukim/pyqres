"""Experiment, sweep, plotting, and CLI helpers."""

from pyqres.dim import (
    ConfigurableSweep,
    LineMetricSpec,
    MemoryObservableStreamingReservoir,
    SweepExperiment,
    build_sweep,
    run_standard_analysis_sweep,
    save_experiment_table,
    save_line_metric_plot,
    sweep_values_from_cfg,
)

from . import data, readout
from .common import (
    build_memory_observable_reservoir,
    build_model,
    dataclass_from_config,
    dataset_from_config,
    readout_from_config,
    reservoir_spec_from_config,
    resolve_output_dir,
    run_experiment_from_config,
    save_raw_dataset,
    select_observable_specs,
    to_builtin,
)
from .datasets import Dataset, DatasetSplit
from .metrics import Metric, mse, negative_rmse, r2, resolve_metrics, rmse
from .readout import ReadoutModel, Ridge
from .runner import Experiment, ExperimentResult, Sweep, SweepResult

__all__ = [
    "ConfigurableSweep",
    "Dataset",
    "DatasetSplit",
    "Experiment",
    "ExperimentResult",
    "LineMetricSpec",
    "MemoryObservableStreamingReservoir",
    "Metric",
    "ReadoutModel",
    "Ridge",
    "Sweep",
    "SweepExperiment",
    "SweepResult",
    "build_memory_observable_reservoir",
    "build_model",
    "build_sweep",
    "data",
    "dataclass_from_config",
    "dataset_from_config",
    "mse",
    "negative_rmse",
    "r2",
    "readout",
    "readout_from_config",
    "reservoir_spec_from_config",
    "resolve_metrics",
    "resolve_output_dir",
    "rmse",
    "run_experiment_from_config",
    "run_standard_analysis_sweep",
    "save_experiment_table",
    "save_line_metric_plot",
    "save_raw_dataset",
    "select_observable_specs",
    "sweep_values_from_cfg",
    "to_builtin",
]
