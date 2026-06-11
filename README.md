# pyqres

`pyqres` is a task-agnostic quantum reservoir computing library. It provides
shared interfaces for constructing reservoirs, running generic supervised
experiments, simulating exact small systems, building Qiskit circuits, and
performing PTM/Volterra/dimensional analysis.

Benchmark task presets have moved to the separate `pyqres-tasks` package. The
legacy task-integrated code line is preserved on the `release/v0.1.x` branch.

## Design

The package is organized around a few public layers:

- `pyqres.core`: backend-neutral Hamiltonian objects, reservoir specs,
  reservoir construction, fluent reservoir builders, and protocols.
- `pyqres.simulation`: dense exact and finite-shot trajectory reservoirs.
- `pyqres.qiskit`: Qiskit circuit reservoirs.
- `pyqres.dim`: PTM, Volterra, visibility, and sweep-analysis tools.
- `pyqres.experiments`: generic datasets, data builders, readouts, metrics,
  experiment runners, result objects, sweeps, and config-driven execution.
- `pyqres.baselines`: classical model utilities that can be used by external
  tasks or comparison studies.

The main public protocols are re-exported from both `pyqres` and `pyqres.core`:

- `TransformReservoirProtocol`: exposes `transform(inputs) -> features`.
- `StatefulReservoirProtocol`: adds `reset(...)` and `step(u)`.
- `QRCReservoirProtocol`: legacy-compatible `reset`, `step`, and `run`.
- `ChannelReservoirProtocol`: adds channel/PTM access.
- `CircuitReservoirProtocol`: exposes circuit construction.
- `DatasetProtocol`, `ReadoutProtocol`, `SerializableSpecProtocol`, and
  `ExperimentResultProtocol`: contracts used by generic experiments.

For convenience, common APIs are re-exported from the top-level `pyqres`
namespace. The implementation still lives in categorized packages:

- `qres.reservoir(...)` -> `pyqres.core.fluent`
- `qres.compile_reservoir(...)` -> `pyqres.core.builders`
- `qres.ReservoirSpec` -> `pyqres.core.specs`
- `qres.data.*`, `qres.readout.*`, and `qres.Experiment` ->
  `pyqres.experiments`

The central Hamiltonian convention remains:

```text
H(u) = H0 + input_scale * u * H1
```

`H0` and `H1` can be represented as backend-neutral Hamiltonian specs, dense
matrices, sparse matrices, or Qiskit operator objects where supported.

## Installation

For local development:

```bash
python -m pip install -e .
```

Install optional feature groups as needed:

```bash
python -m pip install -e .[simulation]
python -m pip install -e .[qiskit]
python -m pip install -e .[dim]
python -m pip install -e .[experiments]
python -m pip install -e .[all]
python -m pip install -e .[dev]
```

Install benchmark task presets separately:

```bash
python -m pip install -e ../pyqres-tasks
```

## Reservoir-First Workflow

The primary interface is a small Python workflow: build a reservoir, pass data
through it, and optionally fit a readout.

```python
import numpy as np
import pyqres as qres

series = np.sin(np.linspace(0.0, 12.0, 1000))

reservoir = (
    qres.reservoir("ising")
    .memory_qubits(5)
    .readout_qubits(2)
    .input("Z", site=0, strength=1.2)
    .evolution(tau=0.6)
    .observables("rich", count=8)
    .backend("exact")
)

dataset = qres.data.timeseries(series, target_horizon=1).split(
    washout=100,
    train=600,
    test=250,
)

result = qres.Experiment(
    reservoir=reservoir,
    dataset=dataset,
    readout=qres.readout.Ridge(l2=1e-6),
    metrics=["r2", "mse"],
).run()

print(result.metrics)
```

For direct supervised arrays:

```python
inputs = np.linspace(-1.0, 1.0, 200)
targets = inputs**2

dataset = qres.data.arrays(inputs, targets).split(
    washout=20,
    train=120,
    test=40,
)

reservoir = (
    qres.reservoir("ising")
    .memory_qubits(2)
    .readout_qubits(1)
    .ancilla_probabilities()
    .backend("exact")
)

result = qres.Experiment(reservoir, dataset, readout=qres.readout.Ridge()).run()
```

The same `Dataset` and `Experiment` objects work with custom user tasks,
`pyqres-tasks` presets, or data loaded from files.

## Config-Driven Runs

Core `pyqres` can run a generic supervised experiment from YAML without any
task-specific code:

```yaml
dataset:
  source: npz
  path: dataset.npz
  split:
    washout: 20
    train: 120
    test: 40

reservoir:
  family: ising
  n_system: 2
  n_ancilla: 1
  tau: 0.6

backend: exact
readout:
  kind: ridge
  l2: 1.0e-6
metrics: [r2, mse]
paths:
  output_dir: outputs/example
  timestamped: true
```

Run it with:

```bash
pyqres-run experiment.yaml
```

The runner writes `metrics.json`, `metadata.json`, and `arrays.npz` containing
features and predictions.

The same configuration can be submitted directly as a Python dictionary:

```python
from pyqres.experiments import run_experiment_from_config

result = run_experiment_from_config({
    "dataset": {"source": "npz", "path": "dataset.npz"},
    "reservoir": {"family": "ising", "n_system": 2, "n_ancilla": 1, "tau": 0.6},
    "backend": "exact",
    "readout": {"kind": "ridge", "l2": 1.0e-6},
    "metrics": ["r2", "mse"],
    "paths": {"output_dir": "outputs/example", "timestamped": False},
})
```

## Reservoir Specs

The fluent API builds `ReservoirSpec` objects internally. For lower-level
workflows, specs can still be created and compiled directly:

```python
spec = qres.ReservoirSpec(family="ising", n_system=3, n_ancilla=1, tau=0.8)

exact = qres.compile_reservoir(spec, backend="exact")
hardware_like = qres.compile_reservoir(spec, backend="hardware_trajectory")
memory_obs = qres.compile_reservoir(
    spec.with_updates(
        readout=qres.ReadoutSpec(
            mode="memory_observables",
            observables="rich",
            count=8,
            init_state="zero",
        )
    ),
    backend="memory_observable",
)
```

Low-level APIs remain available for cases that need direct control over
Hamiltonians, measurement protocols, Qiskit transpilation, or Volterra analysis.

## Task Presets

Task presets now live outside core `pyqres`:

```python
from pyqres_tasks import MackeyGlassConfig, mackey_glass_dataset

dataset = mackey_glass_dataset(
    MackeyGlassConfig(T_total=600, washout=50, train_len=350, test_len=150)
)
```

This keeps `pyqres` focused on reservoir construction and analysis while still
allowing conventional benchmarks to compose with the same generic experiment API.
