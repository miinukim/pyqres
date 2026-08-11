# pyqres Architecture

This document is the shortest route to the right part of the library. The
public API remains centered on `import pyqres as qres`; the folders below group
implementation details by responsibility.

## Package Map

```text
src/pyqres/
├── core/                 configuration, contracts, factories, and compilation
│   ├── specs.py          user-facing serializable reservoir specs
│   ├── factory.py        dictionary -> ReservoirSpec
│   ├── builders.py       ReservoirSpec -> executable backend
│   ├── control.py        measurement and feedback control
│   ├── protocols/        structural interfaces, grouped by consumer
│   └── reservoir_params/ Hamiltonian operators, specs, and generators
├── simulation/           dense exact and finite-shot reservoir backends
├── qiskit/               Qiskit circuit construction and execution
├── dim/                  PTM, Volterra, visibility, and dimension analysis
│   └── model/            shared channel engine plus model families
├── prethermal_shadow/    global-Floquet partial-shadow reservoir
│   └── dynamics/         operators, parameters, Hamiltonians, and states
├── experiments/          datasets, readouts, metrics, runners, and CLI
├── baselines/            classical comparison models
├── presets.py            named reservoir configurations
└── utils/                small internal numerical helpers
```

The separate `src/qres/` package is only a short import alias for `pyqres`.
Benchmark datasets and task-specific workflows belong in `pyqres-tasks`, not
in this repository's core package.

## Runtime Flow

```text
dictionary / ReservoirSpec
          │
          ▼
  core.factory + core.builders
          │
          ├── simulation backend
          ├── dimension-model backend
          ├── Qiskit backend
          ├── prethermal-shadow backend
          └── caller-provided object
                       │
                       ▼
                 feature matrix
                       │
                       ▼
       experiments.Dataset + Ridge + metrics
                       │
                       ▼
                ExperimentResult
```

Dependencies should point down this flow. Backend implementations may consume
core specs and contracts; core must not import task-specific datasets or
experiment scripts.

## Focused Internal Packages

Some historical modules became packages so related code can be found without
searching a single large file. Their original import paths are compatibility
facades.

### `core.protocols`

- `types.py`: shared aliases such as `InputSequence` and `FeatureMatrix`
- `reservoirs.py`: executable reservoir and Qiskit artifact contracts
- `specifications.py`: serializable spec contracts
- `construction.py`: builder, compiler, factory, and preset contracts
- `experiments.py`: dataset, readout, experiment, sweep, and task contracts

### `core.reservoir_params`

- `operators.py`: Pauli terms and dense/Qiskit conversion
- `specifications.py`: backend-neutral `HamiltonianSpec`
- `parameters.py`: `ReservoirParams` and named Hamiltonian generators

### `dim.model`

- `base.py`: memory channel, Kraus, PTM, caching, and fixed-point machinery
- `ising.py`: Ising parameters and model
- `random_pauli.py`: random Pauli-circuit parameters and model
- `syk.py`: number-conserving SYK parameters and model

### `prethermal_shadow.dynamics`

- `operators.py`: Pauli parsing and symbolic operator construction
- `parameters.py`: qubit partitions, topology, validation, and random draws
- `hamiltonians.py`: Floquet Hamiltonians and step unitaries
- `states.py`: density states, tensor reordering, and partial traces

## API Compatibility Rule

Public imports continue to use the historical facade paths:

```python
from pyqres.core.protocols import DatasetProtocol
from pyqres.core.reservoir_params import HamiltonianSpec
from pyqres.dim.model import IsingReservoirModel
from pyqres.prethermal_shadow.dynamics import partial_trace_qubits
```

The focused submodules are available for maintainers, but user code should
prefer package facades or the top-level `pyqres` namespace. When moving an
implementation, keep its facade export, signature, and class module metadata
stable and add a compatibility assertion to the tests.

## Where New Code Belongs

- New backend: its own module under `simulation/`, `qiskit/`, or a new backend
  package, plus one compiler branch in `core/builders.py`.
- New reservoir family for dimension analysis: a focused module under
  `dim/model/` and exports from `dim/model/__init__.py` and `dim/__init__.py`.
- New experiment metric or readout: `experiments/metrics.py` or
  `experiments/readout.py`.
- New named configuration: `presets.py`; do not embed it in the compiler.
- New benchmark dataset or task runner: the external `pyqres-tasks` package.

For a walkthrough of user-facing construction and execution, continue with
[`user_guide.md`](user_guide.md).
