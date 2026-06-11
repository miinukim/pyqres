# pyqres

`pyqres` is a task-agnostic quantum reservoir computing library. It provides
shared interfaces for constructing reservoirs, running generic supervised
experiments, simulating exact small systems, building Qiskit circuits, and
performing PTM/Volterra/dimensional analysis.

Benchmark task presets have moved to the separate `pyqres-tasks` package. The
legacy task-integrated code line is preserved on the `release/v0.1.x` branch.

## Design

The core package is organized around a few public layers:

- `pyqres.core`: backend-neutral Hamiltonian and protocol objects.
- `pyqres.simulation`: dense exact and finite-shot trajectory reservoirs.
- `pyqres.qiskit`: Qiskit circuit reservoirs.
- `pyqres.dim`: PTM, Volterra, visibility, and sweep-analysis tools.
- `pyqres.datasets`: generic array/time-series dataset containers.
- `pyqres.readout`: generic readout models such as ridge regression.
- `pyqres.experiment`: task-agnostic experiment and sweep runners.
- `pyqres.specs` and `pyqres.builders`: serializable reservoir specs and
  backend compilation helpers.
- `pyqres.baselines`: classical model utilities that can be used by external
  tasks or comparison studies.

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

## Generic Workflow

Use `pyqres` with arbitrary arrays or externally generated datasets:

```python
import numpy as np
import pyqres as qres

inputs = np.linspace(-1.0, 1.0, 200)
targets = np.roll(inputs, -1)

dataset = qres.Dataset.from_arrays(
    inputs[:-1],
    targets[:-1],
    washout=20,
    train=120,
    test=40,
)

spec = qres.ReservoirSpec(
    family="ising",
    n_system=2,
    n_ancilla=1,
    tau=0.6,
    input_scale=1.0,
)

reservoir = qres.compile_reservoir(spec, backend="exact")
result = qres.Experiment(
    reservoir=reservoir,
    dataset=dataset,
    readout=qres.Ridge(l2=1e-6),
    metrics=["r2", "mse"],
).run()

print(result.metrics)
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

## Reservoir Specs

Reservoir construction can be described with `ReservoirSpec` and compiled into
different backends:

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
