"""Public API for PTM, Volterra, visibility, and sweep-analysis tools."""

from .analysis import (
    DenseVolterraAnalyzer,
    PTMAffineExpansion,
    TruncatedVolterraGenerator,
    VolterraAnalyzer,
    VolterraResult,
)
from .isotropy import (
    CompressedVisibilityDiagnostics,
    compressed_visibility_diagnostics,
    compressed_visibility_metrics,
)
from .model import (
    FloquetIsingReservoirBase,
    FloquetIsingReservoirParameters,
    HaarRandomReservoirModel,
    HaarRandomReservoirParameters,
    IsingReservoirModel,
    IsingReservoirParameters,
    ReservoirBase,
    SYKReservoirModel,
    SYKReservoirParameters,
    ThreeStepFloquetIsingReservoirModel,
    TwoStepFloquetIsingReservoirModel,
)
try:  # pragma: no cover
    from .qrclib_model import ExactQRCModel, ExactQRCModelConfig, QRCLibExactReservoirModel
except Exception:  # pragma: no cover
    # Keep the package importable even when the optional pyqres exact backend is absent.
    ExactQRCModel = None  # type: ignore
    ExactQRCModelConfig = None  # type: ignore
    QRCLibExactReservoirModel = None  # type: ignore
try:  # pragma: no cover
    from .streaming import MemoryObservableStreamingReservoir, SharedExactStreamingReservoir
except Exception:  # pragma: no cover
    # Streaming depends on the same external backend and should fail soft as well.
    MemoryObservableStreamingReservoir = None  # type: ignore
    SharedExactStreamingReservoir = None  # type: ignore
try:  # pragma: no cover
    from .sweep import ConfigurableSweep, SweepExperiment, build_sweep
except Exception:  # pragma: no cover
    # Sweep helpers pull in plotting/dataframe deps that are useful but optional for core models.
    ConfigurableSweep = None  # type: ignore
    SweepExperiment = None  # type: ignore
    build_sweep = None  # type: ignore
try:  # pragma: no cover
    from .experiment_utils import (
        LineMetricSpec,
        run_standard_analysis_sweep,
        save_experiment_table,
        save_line_metric_plot,
        sweep_values_from_cfg,
    )
except Exception:  # pragma: no cover
    # Experiment helpers depend on plotting/dataframe config just like the sweep helpers.
    LineMetricSpec = None  # type: ignore
    run_standard_analysis_sweep = None  # type: ignore
    save_experiment_table = None  # type: ignore
    save_line_metric_plot = None  # type: ignore
    sweep_values_from_cfg = None  # type: ignore

# Re-export the main analysis, model, and sweep entry points as the package public API.
__all__ = [
    "DenseVolterraAnalyzer",
    "PTMAffineExpansion",
    "TruncatedVolterraGenerator",
    "VolterraAnalyzer",
    "VolterraResult",
    "CompressedVisibilityDiagnostics",
    "compressed_visibility_diagnostics",
    "compressed_visibility_metrics",
    "ReservoirBase",
    "IsingReservoirModel",
    "IsingReservoirParameters",
    "HaarRandomReservoirModel",
    "HaarRandomReservoirParameters",
    "FloquetIsingReservoirBase",
    "FloquetIsingReservoirParameters",
    "TwoStepFloquetIsingReservoirModel",
    "ThreeStepFloquetIsingReservoirModel",
    "SYKReservoirModel",
    "SYKReservoirParameters",
    "QRCLibExactReservoirModel",
    "MemoryObservableStreamingReservoir",
    "SharedExactStreamingReservoir",
    "ExactQRCModel",
    "ExactQRCModelConfig",
    "ConfigurableSweep",
    "SweepExperiment",
    "build_sweep",
    "LineMetricSpec",
    "run_standard_analysis_sweep",
    "save_experiment_table",
    "save_line_metric_plot",
    "sweep_values_from_cfg",
]
