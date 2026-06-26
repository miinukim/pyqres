# pyqres

`pyqres` is a task-agnostic quantum reservoir computing library. Use it to build a reservoir, stream your own data through it, fit a classical readout, and inspect the resulting features or metrics. Benchmark datasets and task presets are intentionally kept outside the core package in `pyqres-tasks`.

The most common workflow is plain Python:

```python
import numpy as np
import pyqres as qres

series = np.sin(np.linspace(0.0, 12.0, 200))

reservoir = qres.qresreservoir.from_dict({
    "preset": "Ising",
    "memory_qubits": 3,
    "readout_qubits": 1,
    "encoding": {"mode": "hamiltonian", "operator": "Z", "targets": [0], "scale": 1.2},
    "dynamics": {"kind": "preset", "name": "Ising", "tau": 0.6},
    "readout": {
        "mode": "memory_observables",
        "observables": {"preset": "rich", "count": 4},
        "include_bias": True,
        "init_state": "zero",
    },
    "backend": "exact",
})

dataset = qres.data.timeseries(series, target_horizon=1).split(
    washout=20,
    train=120,
    test=50,
)

result = qres.Experiment(
    reservoir=reservoir,
    dataset=dataset,
    readout=qres.readout.Ridge(l2=1e-6),
    metrics=["r2", "mse"],
).run()

print(result.metrics)
```

## What You Can Build

A pyqres experiment has four pieces:

- `reservoir`: turns an input stream into a feature matrix.
- `dataset`: holds inputs, targets, and washout/train/test indices.
- `readout`: fits predictions from reservoir features.
- `metrics`: scores train and test predictions.

Reservoirs can come from built-in presets, explicit Hamiltonians, raw Qiskit circuits, or your own Python object. The core rule is simple: if an object exposes `transform(inputs)`, `run_stream(inputs)`, or `run(inputs)`, pyqres can use it in an `Experiment`.

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

## Reservoir Dictionaries

`qres.qresreservoir.from_dict({...})` is the recommended construction API. It accepts a plain dictionary and returns a compiled reservoir.

Minimal preset example:

```python
reservoir = qres.qresreservoir.from_dict({
    "preset": "Ising",
    "memory_qubits": 2,
    "readout_qubits": 1,
    "backend": "exact",
})
```

Explicit Hamiltonian example:

```python
reservoir = qres.qresreservoir.from_dict({
    "memory_qubits": 1,
    "readout_qubits": 1,
    "encoding": {"mode": "hamiltonian", "operator": "Z", "targets": [1], "scale": 0.5},
    "dynamics": {
        "h0_terms": [(1.0, ((0, "X"),))],
        "h1_terms": [(0.5, ((1, "Z"),))],
    },
    "readout": {"mode": "ancilla_probabilities", "include_bias": False},
    "backend": "exact",
})
```

Raw Qiskit circuit example:

```python
from qiskit import QuantumCircuit

circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)

reservoir = qres.qresreservoir.from_dict({
    "memory_qubits": 1,
    "readout_qubits": 1,
    "dynamics": circuit,
    "backend": "qiskit",
})
```

If you need to inspect the normalized spec before compilation:

```python
builder = qres.qresreservoir.builder_from_dict({
    "preset": "Ising",
    "memory_qubits": 1,
    "readout_qubits": 1,
    "backend": "exact",
})
print(builder.spec.to_dict())
reservoir = builder.build()
```

## Data and Experiments

For one-step or multi-step forecasting from a scalar series:

```python
dataset = qres.data.timeseries(series, target_horizon=1).split(
    washout=100,
    train=600,
    test=300,
)
```

For supervised arrays:

```python
dataset = qres.data.arrays(inputs, targets).split(
    washout=20,
    train=120,
    test=40,
)
```

For data saved in `.npz` files:

```python
dataset_builder = qres.data.npz("dataset.npz", inputs_key="inputs", targets_key="targets")
dataset = dataset_builder.split(washout=10, train=100, test=50)
```

Run an experiment:

```python
result = qres.Experiment(
    reservoir=reservoir,
    dataset=dataset,
    readout=qres.readout.Ridge(l2=1e-6),
    metrics=["r2", "mse"],
).run()

result.save("outputs/my_run")
```

`ExperimentResult.save(...)` writes `metrics.json`, `metadata.json`, and `arrays.npz` containing features and predictions.

## Config-Driven Runs

You can also run a generic supervised experiment from YAML:

```yaml
dataset:
  source: timeseries
  series: [0.0, 0.10, 0.21, 0.31, 0.40, 0.48, 0.55, 0.61, 0.66, 0.70, 0.73, 0.75, 0.76]
  target_horizon: 1
  split:
    washout: 2
    train: 6
    test: 4

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
python -m pyqres.experiments.cli experiment.yaml
```

The same structure can be submitted directly as a Python dictionary to `pyqres.experiments.run_experiment_from_config(...)`.

## Option Reference

### `qres.qresreservoir.from_dict(config)`

Accepted top-level reservoir dictionary fields:

| Field | Type | Choices / meaning | Default |
| --- | --- | --- | --- |
| `preset` | `str` | Built-in preset name. Choices: `ising`, `ising.memory_readout`, `random_pauli`, `randompauli`, `random_pauli.memory_readout`, `syk`, `syk.memory_readout`. Case-insensitive in practice. | `ising` |
| `memory_qubits` | `int` | Number of recurrent memory/system qubits. Aliases: `n_memory`, `n_system`, `system_qubits`. | required by most compiled reservoirs |
| `readout_qubits` | `int` | Number of readout/ancilla qubits. Aliases: `n_readout`, `n_ancilla`, `ancilla_qubits`. | required by most compiled reservoirs |
| `seed` | `int` | Random seed for presets/backends. | backend/preset default |
| `tau` | `float` | Evolution time for one reservoir step. Can also appear inside `dynamics`. | `1.0` |
| `encoding` | `dict` or `InputEncodingSpec` | Input encoding metadata. See `encoding` options below. | Hamiltonian encoding with no operator |
| `dynamics` | mapping, pair, circuit, reservoir object, or `DynamicsSpec` | Reservoir dynamics. See dynamics options below. | preset dynamics for `preset` |
| `readout` | `dict` or `ReadoutSpec` | Feature extraction options. See readout options below. | memory observables, `z` |
| `backend` | `str` | `exact`, `dense`, `channel_map`, `memory_observable`, `dim`, `hardware`, `hardware_trajectory`, `qiskit`. | `exact` |
| `model_kwargs` | `dict` | Extra preset/model constructor parameters. Aliases: `model_params`, `model_config`. | `{}` |

Legacy top-level fields such as `input`, `evolution`, `observables`, `hamiltonian`, and `circuit` are intentionally not accepted. Put those concepts under `encoding`, `dynamics`, or `readout`.

### `encoding` Options

`encoding` is normalized by `InputEncodingSpec.from_mapping(...)`.

| Field | Type | Choices / meaning | Default |
| --- | --- | --- | --- |
| `mode` | `str` | Common value: `hamiltonian`. Other values may be used as metadata for custom compilers/reservoirs. | `hamiltonian` |
| `operator` | `str | None` | For built-in Hamiltonian/dimension adapters: `X`, `Y`, or `Z`. Alias: `axis`. | `None` |
| `targets` | sequence of `int` | Target qubit indices. Aliases: `site`, `sites`. | `[]` |
| `scale` | `float` | Input/operator strength. Alias: `strength`. | `1.0` |
| `bias` | `float` | Input bias metadata. | `0.0` |
| `parameters` | `dict` | Extra encoding metadata. Unknown keys in the mapping are moved here. | `{}` |
| `on_memory` | `bool` | Extra key used by dimension presets; maps targets to memory qubits when true. Stored under `parameters`. | preset default |
| `normalization` | `str` | Extra key used by dimension presets. Choices: `none`, `sum`, `sqrt`, `frobenius`, `mean`, `average`. Stored under `parameters`. | preset default |

### `dynamics` Options

The factory infers dynamics from the object you pass:

| Input form | Meaning |
| --- | --- |
| `None` | Use the selected preset. |
| `DynamicsSpec(...)` | Use the spec directly. |
| `{"kind": "preset", "name": "Ising", ...}` | Named preset dynamics. Extra keys become preset/model parameters. Aliases for `name`: `preset`, `family`. |
| `{"kind": "hamiltonian", "parameters": {...}}` | Explicit Hamiltonian dynamics. |
| `{"h0_terms": ..., "h1_terms": ...}` | Explicit Pauli-term Hamiltonian dynamics. |
| `{"h0_matrix": ..., "h1_matrix": ...}` | Explicit matrix/operator Hamiltonian dynamics. Aliases include `h0`, `h1`, `H0`, `H1`, `H0_matrix`, `H1_matrix`, `H0_hamiltonian`, `H1_hamiltonian`. |
| `(H0, H1)` | Two-item Hamiltonian pair. |
| raw Qiskit-like circuit object | Any object with `num_qubits` and `to_instruction`; compiled with `backend="qiskit"`. |
| `{"kind": "circuit", "circuit": circuit, ...}` | Explicit circuit dynamics with extra circuit kwargs. |
| existing reservoir object | Any object exposing `transform`, `run_stream`, `run`, or `step`. |
| `{"kind": "object", "reservoir": obj}` | Explicit existing-reservoir dynamics. |

Built-in preset parameter choices:

| Preset | Useful parameters |
| --- | --- |
| `ising` | `tau`, `seed`, `gx_memory`, `gz_memory`, `jzz_memory`, `jxx_memory`, `jzz_next_nearest`, `gx_readout`, `gz_readout`, `kz_memory_readout`, `input_strength`, `input_axis` (`X`, `Y`, `Z`), `input_on_memory`, `input_site`, `input_sites`, `input_strength_normalization` (`none`, `sum`, `sqrt`, `frobenius`, `mean`, `average`), `periodic_memory_chain`, `reset_to_zero_state`. |
| `random_pauli` / `randompauli` | `depth`, `seed`, `input_qubit`, `encoding_qubits`, `ancilla_state` (`zero`), `input_bias`, `input_scale`. |
| `syk` | `tau`, `j4_strength`, `kappa2_strength`, `seed`, `input_qubit`, `input_bias`, `input_scale`, `input_clip_eps`, `normalize_syk4_by_spectral_norm`, `normalize_syk2_by_spectral_norm`, `reset_to_zero_state`. |

### `readout` Options

`readout.mode` controls the feature rows emitted by the reservoir.

| Field | Type | Choices / meaning | Default |
| --- | --- | --- | --- |
| `mode` | `str` | Memory-observable aliases: `memory_observables`, `observables`. Ancilla-probability aliases: `ancilla`, `ancilla_probs`, `ancilla_probabilities`, `probabilities`. | `memory_observables` in dictionaries, `ancilla_probs` in bare `ReadoutSpec` |
| `include_bias` | `bool` | Prepend a constant `1.0` feature column. | `True` |
| `init_state` | `str` | For memory-observable reservoirs: `zero`, `fixed_point`, `maximally_mixed`. For exact/channel-map reservoirs: `zero`, `maximally_mixed`. | `zero` for memory observables, `maximally_mixed` for ancilla probabilities |
| `shots` | `int` | Number of shots for finite-shot/noisy feature estimates. | `4096` |
| `shot_noise` | `bool` | Alias for `use_shot_noise`. Applies to ancilla-probability readout. | `False` |
| `use_shot_noise` | `bool` | Same as `shot_noise`. | `False` |
| `observables` | `str`, list, or dict | Observable preset/list. Dict form: `{"preset": "rich", "count": 8, "custom": [...]}`. | `z` |
| `count` | `int | None` | Keep only the first `count` resolved observable specs. | `None` |
| `custom` | list of `str` | Extra observable strings such as `Z0`, `X0*Z1`, or SYK number observables such as `N0`. | `[]` |

Observable presets for Ising/RandomPauli-style memory models:

- `z`, `x`, `y`
- `xy`, `zx`, `xyz`
- `zz_pairs`, `xx_pairs`, `nn_pairs`, `pair_xyz`
- `rich`
- `custom`

Observable presets for SYK memory models:

- `occupation`, `occupations`, `number`
- `occupation_pairs`
- `occupation_rich`
- all generic Pauli presets above are also available through the base model path

### Backend Choices

| Backend | What it returns | Notes |
| --- | --- | --- |
| `exact`, `dense`, `channel_map` | `ChannelMapReservoir` | Dense exact deterministic channel-map features. If readout mode is `memory_observables`, `exact`/`dense` are redirected to `memory_observable`. |
| `memory_observable`, `dim` | `MemoryObservableStreamingReservoir` | Dimension-model memory observable features. Supports Ising, RandomPauli, and SYK presets. |
| `hardware`, `hardware_trajectory` | `HardwareTrajectoryReservoir` | Dense finite-shot trajectory emulator. |
| `qiskit` | `QRCReservoir` | Builds Qiskit circuits. Preset Hamiltonians are converted to `SparsePauliOp`; raw circuits are appended directly. |

### Dataset Builder Options

`qres.data.timeseries(series, target_horizon=1, metadata=None)`:

| Parameter | Meaning | Default |
| --- | --- | --- |
| `series` | Scalar sequence used as inputs. Targets are shifted by `target_horizon`. | required |
| `target_horizon` | Positive forecast horizon. | `1` |
| `metadata` | Optional metadata dictionary. | `{}` |
| `.split(washout, train, test)` | Contiguous split lengths. | `washout=0`; `train` and `test` required |

`qres.data.arrays(inputs, targets, metadata=None)`:

| Parameter | Meaning | Default |
| --- | --- | --- |
| `inputs` | Input array. First dimension is sample index. | required |
| `targets` | Target array with same first dimension as `inputs`. | required |
| `metadata` | Optional metadata dictionary. | `{}` |
| `.split(washout, train, test, indices=None)` | Either contiguous split lengths or explicit index mapping with `washout`, `train`, `test`. | `washout=0`; `train` and `test` required |

`qres.data.npz(path, inputs_key="inputs", targets_key="targets", metadata=None)` loads arrays from an `.npz` file and returns the same supervised builder as `arrays(...)`.

Config-driven datasets support:

| `dataset.source` | Required fields | Optional fields |
| --- | --- | --- |
| `arrays` | `inputs`, `targets`, `split` | `metadata` |
| `timeseries` | `series`, `split` | `target_horizon`, `metadata` |
| `npz` | `path` | `inputs_key`, `targets_key`, `split`, `metadata` |

### Readout and Metric Options

`qres.readout.Ridge(l2=1e-6, include_bias=False)`:

| Parameter | Meaning | Default |
| --- | --- | --- |
| `l2` | Ridge regularization strength. | `1e-6` |
| `include_bias` | Add a bias column inside the readout. Usually keep `False` when reservoir features already include bias. | `False` |

Config-driven readouts support:

| Field | Choices / meaning | Default |
| --- | --- | --- |
| `kind` | `ridge` | `ridge` |
| `l2` | Ridge regularization strength. | `1e-6` |
| `include_bias` | Add readout-side bias. | `False` |

Metric names accepted by `Experiment(..., metrics=[...])` and config-driven runs:

- `r2`
- `mse`
- `rmse`
- `negative_rmse`

If `metrics=None`, pyqres uses `r2` and `mse`.

### `Experiment` Options

`qres.Experiment(reservoir, dataset, readout=None, metrics=None, metadata=None)`:

| Parameter | Choices / meaning | Default |
| --- | --- | --- |
| `reservoir` | Any object exposing `transform(inputs)`, `run_stream(inputs)`, or `run(inputs)`. | required |
| `dataset` | A `Dataset` with `inputs`, `targets`, and split indices. | required |
| `readout` | Any object exposing `fit(features, targets)` and `predict(features)`. Common choice: `qres.readout.Ridge(...)`. | `Ridge()` |
| `metrics` | `None`, a list/tuple of metric names, or a mapping of names to callables. Built-in names: `r2`, `mse`, `rmse`, `negative_rmse`. | `None` |
| `metadata` | Optional dictionary stored in the result metadata. | `{}` |

`ExperimentResult.save(outdir)` writes `metrics.json`, `metadata.json`, and `arrays.npz`.

Config-driven `paths` options:

| Field | Choices / meaning | Default |
| --- | --- | --- |
| `output_dir` | Directory where run artifacts are written. | `outputs/pyqres_experiment` |
| `timestamped` | If true, append a timestamp subdirectory. | `True` |

### Low-Level `ReservoirSpec` Options

You can bypass the dictionary factory and build specs directly:

```python
spec = qres.ReservoirSpec(
    family="ising",
    n_system=3,
    n_ancilla=1,
    tau=0.8,
)
reservoir = qres.compile_reservoir(spec, backend="exact")
```

`ReservoirSpec` fields:

| Field | Meaning |
| --- | --- |
| `family` | Human/preset family label. |
| `preset` | Preset name, if any. |
| `source_kind` | `preset`, `hamiltonian`, `circuit`, `object`, or compiler-specific kind. |
| `n_system`, `n_memory` | System/memory qubit count. `system_qubits` resolves from these. |
| `n_ancilla`, `n_readout` | Ancilla/readout qubit count. `ancilla_qubits` resolves from these. |
| `tau` | Evolution time. |
| `input_scale` | Scalar input multiplier for Hamiltonian/Qiskit evolution where applicable. |
| `seed` | Random seed. |
| `encoding` | `InputEncodingSpec`. |
| `dynamics` | `DynamicsSpec`. |
| `readout` | `ReadoutSpec`. |
| `model_kwargs` | Dimension-model/preset kwargs. |
| `hamiltonian_kwargs` | Hamiltonian-generation kwargs. |
| `circuit_kwargs` | Qiskit circuit backend kwargs. |
| `runtime` | Non-serializable runtime objects such as raw circuits or reservoir objects. |

### Qiskit Low-Level Options

At the lower-level Qiskit API:

```python
from qiskit.quantum_info import SparsePauliOp
from pyqres.qiskit import QRCConfig, QRCReservoir

reservoir = QRCReservoir(QRCConfig(
    n_system=1,
    n_ancilla=1,
    reservoir_type="pauli_evolution",
    H0_hamiltonian=SparsePauliOp.from_list([("XI", 1.0)]),
    H1_hamiltonian=SparsePauliOp.from_list([("IZ", 0.5)]),
))
```

`QRCConfig` choice fields:

| Field | Choices |
| --- | --- |
| `reservoir_type` | `pauli_evolution`, `custom_circuit`, `random_cx_rz` |
| `evolution_synthesis` | `default`, `lie_trotter`, `suzuki_trotter` |
| `encoding` | `rz_global`, `rz_per_qubit` |
| `input_map` | `global`, `per_qubit_random_sign` |
| `ancilla_pattern` | `star`, `pairwise` |
| `readout` | `z_local`, `z_local_plus_anc`, `pauli_k_local` |
| `simulator_method` | `automatic`, `density_matrix`, `statevector` |

Other useful `QRCConfig` fields: `n_system`, `n_ancilla`, `reservoir_circuit`, `reservoir_circuit_targets`, `H0_hamiltonian`, `H1_hamiltonian`, `evolution_reps`, `evolution_order`, `evolution_insert_barriers`, `evolution_preserve_order`, `depth_per_step`, `tau`, `seed`, `input_scale`, `use_purification`, `measure_and_reset_ancilla`, `pauli_k`, `include_bias`, `shots`, `transpile_optimization_level`, `noise`, and `control`.

`NoiseConfig` fields: `use_damping`, `dt`, `T1`, `T2`, `use_depolarizing`, `p_depol_1q`, `p_depol_2q`.

### Dense Simulation and Measurement Options

`ExactQRCModelConfig` choice fields:

| Field | Choices |
| --- | --- |
| `input_encoding` | `hamiltonian`, `amplitude`, `unitary` |
| `encoding_register` | `ancilla`, `system`, `full` |
| `amplitude_encoding_style` | `u_sqrt_1_minus_u`, `sqrt_u_sqrt_1_minus_u` |
| `input_unitary_order` | `before`, `after`, `replace` |

Other useful fields: `n_system`, `n_ancilla`, `tau`, `hamiltonian_kind`, `input_scale`, `input_bias`, `encoding_targets`, `amplitude_state_factory`, `input_unitary_factory`, `unitary_cache_size`, `H0_hamiltonian`, `H1_hamiltonian`, `H0_matrix`, `H1_matrix`, `seed`, and `control`.

`MeasurementControlConfig` choice fields:

| Field | Choices |
| --- | --- |
| `measurement_mode` | `projective`, `weak` |
| `post_measurement_mode` | `reset`, `keep` |
| `conditioned_gate` | `none`, `system_x`, `system_rx`, `system_rz`, `ancilla_x`, `ancilla_rx`, `ancilla_rz` |
| `conditioned_gate_condition` | `nonzero`, `all_one` |

Other fields: `measurement_strength` in `[0, 1]`, `conditioned_gate_angle`, and `conditioned_gate_target`.

`ChannelMapReservoirConfig` adds `include_bias`, `use_shot_noise`, `shots`, and `init_state` (`maximally_mixed` or `zero`).

`HardwareTrajectoryReservoirConfig` adds `include_bias`, `shots`, and `init_state` (`zero` or `maximally_mixed`).

## Working With Task Presets

Task presets live outside core `pyqres`:

```python
from pyqres_tasks import MackeyGlassConfig, mackey_glass_dataset

dataset = mackey_glass_dataset(
    MackeyGlassConfig(T_total=600, washout=50, train_len=350, test_len=150)
)
```

This keeps `pyqres` focused on reservoir construction and analysis while still allowing conventional benchmarks to compose with the same generic experiment API.

## Public Modules

For most users, these are the modules to import:

- `pyqres` or `qres`: top-level construction, datasets, experiments, readouts, specs, and protocols.
- `pyqres.presets`: named reservoir spec helpers and preset names.
- `pyqres.qiskit`: direct Qiskit circuit backend control.
- `pyqres.simulation`: dense exact and trajectory backends.
- `pyqres.dim`: PTM, Volterra, visibility, and dimension-analysis tools.
- `pyqres.baselines`: ESN and logistic/softmax classical baselines.
