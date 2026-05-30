# pyqres

`pyqres` is the unified quantum reservoir computing package for this workspace.
It brings together the exact/Qiskit reservoir runtime, benchmark tasks,
classical baselines, and PTM/Volterra dimension analysis that were previously
split across `qrclib` and `qrcdim`.

New code should import from `pyqres.*`. The older package directories can remain
in the workspace during migration, but `pyqres` now has its own implementation
modules under `src/pyqres`.

## Package Layout

```text
pyqres/
  core/         Shared protocols, measurement control, reservoir parameters
  exact/        Dense exact QRC model and exact reservoir frontends
  qiskit/       Qiskit-compatible streaming and noisy reservoirs
  dim/          PTM/Liouville, Volterra, rank, visibility analysis
  tasks/        STM, channel equalization, Mackey-Glass benchmarks
  baselines/    ESN, logistic, and softmax classical baselines
  experiments/  CLI, Hydra configs, sweep helpers, result utilities
  utils/        Internal numerical utilities
```

The intended dependency direction is inward:

```text
pyqres.core
  <- pyqres.exact
  <- pyqres.qiskit
  <- pyqres.dim
  <- pyqres.tasks
  <- pyqres.baselines
  <- pyqres.experiments
```

`core` should stay lightweight. Runtime implementations and analysis code should
depend on it, not the other way around.

## Install

For local development from this directory:

```bash
python -m pip install -e .
```

Install optional groups as needed:

```bash
python -m pip install -e ".[qiskit]"
python -m pip install -e ".[dim,experiments]"
python -m pip install -e ".[all]"
```

The optional groups are organized by use case:

- `exact` - dense exact simulation dependencies
- `qiskit` - Qiskit and Qiskit Aer execution
- `dim` - dimension-analysis dependencies
- `tasks` - benchmark/task dependencies
- `experiments` - Hydra, pandas, and plotting helpers
- `all` - full local research environment
- `dev` - test and lint tooling

## `pyqres.core`

`pyqres.core` contains shared abstractions that other subpackages use.

Files:

- `core/protocols.py` - minimal reservoir protocols and result containers
- `core/control.py` - measurement, reset, weak-measurement, and feedback helpers
- `core/reservoir_params.py` - Hamiltonian parameter and coupling generators

Important exports:

```python
from pyqres.core import (
    QRCReservoirProtocol,
    ChannelReservoirProtocol,
    CircuitReservoirProtocol,
    ReservoirStepResult,
    ReservoirRunResult,
    MeasurementControlConfig,
    ReservoirParams,
)
```

Use this layer when adding new reservoir implementations that should be usable
by tasks or analysis code without binding them to one backend.

## `pyqres.exact`

`pyqres.exact` contains the dense exact reservoir implementation and exact
frontends.

Files:

- `exact/exact_qrc.py` - `ExactQRCModel` and `ExactQRCModelConfig`
- `exact/channel_map.py` - deterministic channel-map reservoir features
- `exact/hardware.py` - sampled hardware-trajectory-style reservoir

Important exports:

```python
from pyqres.exact import (
    ExactQRCModel,
    ExactQRCModelConfig,
    ChannelMapReservoir,
    ChannelMapReservoirConfig,
    HardwareTrajectoryReservoir,
    HardwareTrajectoryReservoirConfig,
)
```

Typical use:

```python
import numpy as np

from pyqres.exact import ChannelMapReservoir, ChannelMapReservoirConfig

cfg = ChannelMapReservoirConfig(
    n_system=2,
    n_ancilla=1,
    tau=0.6,
    input_scale=1.0,
    seed=1,
)
reservoir = ChannelMapReservoir(cfg)
features = reservoir.run(np.linspace(-1.0, 1.0, 20))
```

Use `ExactQRCModel` directly when you need channel-level access, dense unitaries,
or PTM-compatible memory-channel operations.

## `pyqres.qiskit`

`pyqres.qiskit` contains the circuit/backend-facing reservoir implementation.

Files:

- `qiskit/config.py` - Qiskit reservoir and noise configuration dataclasses
- `qiskit/reservoir.py` - streaming Qiskit reservoir implementations

Important exports:

```python
from pyqres.qiskit import (
    QRCConfig,
    QRCReservoir,
    NISQRCConfig,
    NISQReservoir,
    NoiseConfig,
)
```

This layer is for circuit-style reservoir execution, Aer simulation, and noisy
NISQ-style experiments. It should remain backend-oriented and should not import
dimension-analysis code.

## `pyqres.dim`

`pyqres.dim` contains the PTM/Liouville and Volterra-dimension analysis code.

Files:

- `dim/pauli.py` - Pauli basis and Pauli-string utilities
- `dim/linalg_utils.py` - PTM coordinates, ranks, null spaces, derivatives
- `dim/model.py` - Ising, Floquet Ising, Haar-random, and SYK reservoir models
- `dim/analysis.py` - affine PTM expansions and Volterra analyzers
- `dim/isotropy.py` - compressed visibility projector diagnostics
- `dim/qrclib_model.py` - wrapper from the exact pyqres core into dim analysis
- `dim/streaming.py` - task-side streaming adapter over the exact core
- `dim/sweep.py` - configurable sweep machinery
- `dim/experiment_utils.py` - reusable experiment table/plot helpers

Important exports:

```python
from pyqres.dim import (
    ReservoirBase,
    IsingReservoirModel,
    IsingReservoirParameters,
    QRCLibExactReservoirModel,
    IsingVolterraAnalyzer,
    ReducedVolterraAnalyzer,
    TruncatedVolterraGenerator,
    compressed_visibility_diagnostics,
    compressed_visibility_metrics,
)
```

Use this layer for questions like:

- What is the latent Volterra span dimension?
- What is the visible Volterra rank?
- Which latent directions are invisible to a chosen readout?
- How isotropic is the readout-visible projector on the latent span?

Example bridge from the exact runtime into dimension analysis:

```python
from pyqres.dim import QRCLibExactReservoirModel
from pyqres.exact import ExactQRCModelConfig

cfg = ExactQRCModelConfig(n_system=2, n_ancilla=1, seed=1)
model = QRCLibExactReservoirModel(config=cfg)
ptm = model.ptm(0.0)
```

## `pyqres.tasks`

`pyqres.tasks` contains benchmark datasets and task runners.

Files:

- `tasks/stm.py` - short-term memory benchmark
- `tasks/channel_equalization.py` - nonlinear channel equalization tasks
- `tasks/mackey_glass.py` - Mackey-Glass time-series forecasting

Important exports:

```python
from pyqres.tasks import (
    STMConfig,
    STMTaskRunner,
    ChannelEqualizationConfig,
    ChannelEqualizationDatasetConfig,
    ChannelEqualizationTaskRunner,
    generate_channel_equalization_data,
    generate_channel_equalization_dataset,
    collect_channel_equalization_reservoir_features,
    MackeyGlassConfig,
    MackeyGlassTaskRunner,
    generate_mackey_glass_series,
)
```

Task runners expect a reservoir object with a `reset()` method and a `run()` or
`step()` interface compatible with the task protocol.

## `pyqres.baselines`

`pyqres.baselines` contains classical baselines and readout models used for
comparison.

Files:

- `baselines/esn.py` - echo-state network implementation and benchmark helpers
- `baselines/classical.py` - logistic equalizer and multiclass softmax readout

Important exports:

```python
from pyqres.baselines import (
    ESNConfig,
    EchoStateNetwork,
    run_stm_esn,
    run_channel_equalization_esn,
    LogisticEqualizerConfig,
    SoftmaxReadoutConfig,
    fit_softmax_readout,
    predict_softmax_readout,
)
```

Use this layer to compare QRC reservoirs against classical recurrent or linear
readout baselines.

## `pyqres.experiments`

`pyqres.experiments` contains runnable experiment entry points and sweep helpers.

Files:

- `experiments/cli.py` - Hydra-driven CLI entry points and benchmark runners
- `experiments/conf/` - default Hydra configuration tree
- `dim/sweep.py` and `dim/experiment_utils.py` - imported through
  `pyqres.experiments` for sweep-oriented workflows

Console scripts from `pyproject.toml`:

```bash
pyqres-run
pyqres-stm-demo
pyqres-stm-hydra
pyqres-channel-eq-benchmark
```

Useful imports:

```python
from pyqres.experiments import (
    build_sweep,
    ConfigurableSweep,
    SweepExperiment,
    run_standard_analysis_sweep,
    save_experiment_table,
    save_line_metric_plot,
)
```

## Workflow Examples

Run an exact reservoir on a benchmark:

```python
from pyqres.exact import ChannelMapReservoir, ChannelMapReservoirConfig
from pyqres.tasks import STMConfig, STMTaskRunner

reservoir = ChannelMapReservoir(ChannelMapReservoirConfig(n_system=2, n_ancilla=1))
task = STMConfig(T_total=300, washout=50, train_len=150, test_len=75)
scores = STMTaskRunner(reservoir, task).run()
```

Fit a softmax readout on collected features:

```python
from pyqres.baselines import SoftmaxReadoutConfig, fit_softmax_readout, predict_softmax_readout

model = fit_softmax_readout(X_train, y_train, SoftmaxReadoutConfig())
predictions = predict_softmax_readout(X_test, model)
```

Analyze a reservoir through PTM/Volterra tools:

```python
from pyqres.dim import QRCLibExactReservoirModel, IsingVolterraAnalyzer
from pyqres.exact import ExactQRCModelConfig

model = QRCLibExactReservoirModel(
    config=ExactQRCModelConfig(n_system=2, n_ancilla=1, seed=1)
)
analyzer = IsingVolterraAnalyzer(model)
```

## Development Checks

Run the current smoke tests:

```bash
python -m pytest -q
```

Compile the package:

```bash
python -m compileall -q src/pyqres
```

The smoke tests verify the main public imports and a small exact/dimension
bridge path.
