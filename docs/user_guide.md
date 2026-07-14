# pyqres User Guide For Understanding The Code

This guide is for developers and advanced users who want to understand how
`pyqres` is put together without reading every file from top to bottom.

The codebase is intentionally split into two ideas:

- **Core pyqres is task-agnostic.** It knows how to construct reservoirs, run
  input streams through them, fit readouts, and report metrics.
- **Tasks live outside the core.** Benchmark generators such as Mackey-Glass
  belong in `pyqres-tasks` and return ordinary `pyqres.Dataset` objects.

If you keep that separation in mind, the package becomes much easier to read.

## The Mental Model

A normal pyqres workflow looks like this:

```text
plain Python config
    -> qresreservoir.from_dict(...)
    -> ReservoirSpec
    -> compile_reservoir(...)
    -> backend reservoir object
    -> qres.run(reservoir, inputs)
    -> feature matrix
    -> Experiment(...).run()
    -> metrics, predictions, saved arrays
```

There are four user-facing concepts:

- **Reservoir**: converts an input sequence into a feature matrix.
- **Dataset**: stores inputs, targets, and washout/train/test indices.
- **Readout**: fits a classical model from features to targets.
- **Experiment**: runs the reservoir, fits the readout, and scores metrics.

The most important thing is that reservoirs are duck-typed. A reservoir can be
any object exposing one of these methods:

```python
run(inputs)
run_stream(inputs)
transform(inputs)
```

The helper `qres.run(...)` tries those methods in that order.

## Where To Start Reading

Start with these files in this order:

1. `src/pyqres/__init__.py`

   This is the public namespace. It tells you what users are expected to import:
   `qresreservoir`, `Experiment`, `Dataset`, `Ridge`, `compile_reservoir`,
   `run`, specs, protocols, and subpackages.

2. `src/pyqres/core/factory.py`

   This is the dictionary-first construction layer. It accepts user dictionaries
   and turns them into structured specs.

3. `src/pyqres/core/specs.py`

   This defines the immutable configuration objects:
   `InputEncodingSpec`, `DynamicsSpec`, `ReadoutSpec`, and `ReservoirSpec`.

4. `src/pyqres/core/builders.py`

   This is where specs become executable reservoirs. The central function is
   `compile_reservoir(...)`.

5. `src/pyqres/experiments/runner.py`

   This shows how reservoirs, datasets, readouts, and metrics are combined.

After that, jump to the backend you care about:

- Dense/channel simulation: `src/pyqres/simulation/`
- Memory-observable/dimension models: `src/pyqres/dim/`
- Qiskit circuits and GPU/MPS simulation: `src/pyqres/qiskit/`
- Built-in presets: `src/pyqres/presets.py`
- Classical baselines: `src/pyqres/baselines/`

## Public API Versus Internals

Most users should interact with:

```python
import pyqres as qres

reservoir = qres.qresreservoir.from_dict({...})
features = qres.run(reservoir, inputs)
result = qres.Experiment(reservoir, dataset, readout=qres.Ridge()).run()
```

The internal code is organized to support that surface.

Avoid making new user workflows depend directly on low-level classes unless the
user explicitly needs backend control. For example, direct `QRCConfig` and
`QRCReservoir` construction is useful for Qiskit-specific experiments, but
ordinary workflows should go through `qresreservoir.from_dict(...)`.

## How Reservoir Construction Works

The entry point is:

```python
qres.qresreservoir.from_dict(config)
```

That calls:

```text
qresreservoir.builder_from_dict(config)
    -> normalize dimensions, seed, backend
    -> normalize encoding
    -> infer dynamics
    -> normalize readout
    -> store qiskit/simulator options
    -> build ReservoirSpec
    -> compile_reservoir(spec, backend)
```

The dictionary parser lives in `core/factory.py`.

Important behavior:

- `memory_qubits`, `n_memory`, `n_system`, and `system_qubits` are aliases.
- `readout_qubits`, `n_readout`, `n_ancilla`, and `ancilla_qubits` are aliases.
- Raw Qiskit circuits are detected by `num_qubits` and `to_instruction`.
- Existing reservoir objects are detected by `run`, `run_stream`, `transform`,
  or `step`.
- Qiskit simulator options live under `qiskit`, `qiskit_kwargs`, or
  `simulator`, not inside Hamiltonian or preset parameters.

Example:

```python
reservoir = qres.qresreservoir.from_dict({
    "preset": "Ising",
    "memory_qubits": 4,
    "readout_qubits": 2,
    "tau": 1.2,
    "dynamics": {
        "kind": "preset",
        "name": "Ising",
        "input_strength": 0.8,
    },
    "readout": {
        "mode": "memory_observables",
        "observables": "rich",
        "count": 12,
    },
    "backend": "memory_observable",
})
```

To inspect the normalized spec before compilation:

```python
builder = qres.qresreservoir.builder_from_dict({...})
print(builder.spec.to_dict())
reservoir = builder.build()
```

## Specs: The Shared Language

Specs are defined in `core/specs.py`. They are small dataclasses that let the
front-end factory and back-end compilers communicate without hard-coding one
task or one reservoir family.

Use these when you want reproducible construction:

- `InputEncodingSpec`: describes how input values are encoded.
- `DynamicsSpec`: describes reservoir dynamics.
- `ReadoutSpec`: describes feature extraction.
- `ReservoirSpec`: combines dimensions, backend-facing kwargs, dynamics,
  readout, and runtime objects.

`ReservoirSpec` deliberately has separate fields for:

- `model_kwargs`: preset/model parameters.
- `hamiltonian_kwargs`: explicit Hamiltonian construction parameters.
- `circuit_kwargs`: circuit-backend construction parameters.
- `qiskit_kwargs`: Qiskit/Aer simulator options.
- `runtime`: non-serializable Python objects such as raw circuits or existing
  reservoir objects.

This separation is important. It prevents simulator choices such as
`simulator_device="GPU"` from being confused with physical Hamiltonian
parameters.

## Dynamics: Presets Are Helpers, Not The Core

The core accepts several forms of dynamics:

```python
{"kind": "preset", "name": "Ising", ...}
{"kind": "hamiltonian", "h0": H0, "h1": H1}
{"h0_terms": ..., "h1_terms": ...}
(H0, H1)
raw_qiskit_circuit
existing_reservoir_object
```

Presets such as Ising and RandomPauli live in `presets.py`. They are convenience
adapters that produce generic lower-level artifacts. The core should not assume
that every reservoir is Ising-like.

For Qiskit Hamiltonian evolution:

- explicit Hamiltonians and preset-generated Hamiltonians use the same path
- the core converts them to Qiskit-native `SparsePauliOp`
- the Qiskit backend applies them with `PauliEvolutionGate`

The relevant helper is:

```python
build_qiskit_hamiltonian_artifacts(spec)
```

in `core/builders.py`.

## Backend Selection

`compile_reservoir(spec, backend)` lives in `core/builders.py`.

It chooses one of these backend families:

- `memory_observable` or `dim`

  Uses `MemoryObservableStreamingReservoir` from `pyqres.dim`. This is the
  backend that performed well on the Mackey-Glass example. It is CPU dense
  linear algebra today.

- `exact`, `dense`, or `channel_map`

  Uses dense channel-map simulation from `pyqres.simulation`.

- `hardware` or `hardware_trajectory`

  Uses finite-shot trajectory emulation from `pyqres.simulation`.

- `qiskit`

  Uses `QRCReservoir` from `pyqres.qiskit`. It can consume explicit Qiskit
  circuits or Hamiltonians converted to `SparsePauliOp`.

- `object`

  Returns the user-provided reservoir object directly.

Backend choice determines the runtime behavior, not the user task. The same
Mackey-Glass dataset can be used with a dense reservoir, a Qiskit reservoir, or
a custom Python reservoir.

## Custom Input Encoding And Measurement Control

The high-level dictionary API is intentionally centered on portable reservoir
construction: presets, explicit Hamiltonians, raw Qiskit circuits, and existing
reservoir objects. Some dense-simulation controls are lower-level because they
are specific to the exact simulation backend.

Use direct simulation config classes when you need to customize:

- input encoding mode: `hamiltonian`, `amplitude`, or `unitary`
- input target register: `system`, `ancilla`, or `full`
- amplitude state preparation or a custom input-unitary factory
- projective versus weak ancilla measurement
- post-measurement reset versus keep behavior
- measurement-conditioned feedback gates

The relevant files are:

```text
src/pyqres/simulation/exact_qrc.py
src/pyqres/simulation/channel_map.py
src/pyqres/core/control.py
src/pyqres/core/reservoir_params.py
examples/custom_reservoir_input_measurement.py
```

The important semantic split is:

```text
input_encoding="hamiltonian":
    U(u) = exp(-i tau (H0 + input_scale * u * H1))

input_encoding="amplitude" or "unitary":
    U(u) = exp(-i tau H0) composed with an input-dependent encoding unitary
```

So `H1` controls the input only in Hamiltonian-modulation mode. For amplitude
or unitary input encoding, `H0` supplies the fixed reservoir dynamics and the
input enters through the configured encoding unitary.

The direct dense-simulation path still composes with the generic experiment
API:

```python
from pyqres.core.control import MeasurementControlConfig
from pyqres.core.reservoir_params import ReservoirParams
from pyqres.simulation import ChannelMapReservoir, ChannelMapReservoirConfig

hamiltonian = ReservoirParams.from_pauli_terms(
    n_system=1,
    n_ancilla=1,
    h0_terms=[(0.45, ((0, "X"),)), (0.70, ((0, "Z"), (1, "Z")))],
    h1_terms=[],
    tau=0.8,
).generate()

control = MeasurementControlConfig(
    measurement_mode="weak",
    measurement_strength=0.65,
    post_measurement_mode="reset",
    conditioned_gate="system_rz",
)

reservoir = ChannelMapReservoir(ChannelMapReservoirConfig(
    n_system=1,
    n_ancilla=1,
    tau=hamiltonian["tau"],
    H0_hamiltonian=hamiltonian["H0_hamiltonian"],
    H1_hamiltonian=hamiltonian["H1_hamiltonian"],
    input_encoding="amplitude",
    input_scale=0.5,
    input_bias=0.5,
    encoding_register="ancilla",
    encoding_targets=(0,),
    control=control,
))

result = qres.Experiment(reservoir, dataset, readout=qres.Ridge()).run()
```

## Prethermal Shadow Reservoir

The prethermal shadow reservoir is part of the standard pyqres reservoir
construction path. It has dense global-Floquet dynamics, a trace/reset memory
channel, and partial-shadow feature reconstruction, while still satisfying the
generic pyqres reservoir contract:

```python
run_stream(inputs) -> feature_matrix
```

The recommended construction path is `qresreservoir.from_dict`:

```python
reservoir = qres.qresreservoir.from_dict({
    "preset": "prethermal_shadow",
    "memory_qubits": 4,
    "readout_qubits": 2,
    "backend": "exact",
    "floquet": {
        "omega": 12.0,
        "n_cycles_per_step": 4,
        "seed": 0,
    },
    "encoding": {
        "mode": "prethermal_shadow",
        "input_qubits": "memory",
        "axis": "y",
        "scale": 0.12,
        "seed": 1,
    },
    "shadow": {
        "pauli_k": 2,
        "shots": 2048,
        "measurement_type": "weak",
        "weak_strength": 0.5,
        "seed": 2,
    },
    "reset": {"reset_state": "zero"},
})
```

The lower-level classes remain available for advanced diagnostics and custom
construction. The implementation is split by responsibility:

```text
src/pyqres/prethermal_shadow/config.py
src/pyqres/prethermal_shadow/dynamics.py
src/pyqres/prethermal_shadow/shadows.py
src/pyqres/prethermal_shadow/reservoir.py
src/pyqres/prethermal_shadow/diagnostics.py
```

Key runtime semantics:

- Memory qubits are laid out first and persist across the entire input stream.
- Readout qubits are the trailing subset. The global Floquet Hamiltonian acts
  on memory and readout qubits together; there is no explicit transducer block.
- Features are computed from the pre-reset readout marginal, either as exact
  Pauli expectations or as projective/weak local Pauli shadow estimates.
- After feature extraction, readout is traced out and reset to `zero` or `plus`
  for the next step. The default reset state is `zero`.
- Feature labels are all non-identity readout Pauli strings up to `pauli_k`,
  with optional leading `bias`.

The customization surface is deliberately modular:

- Use `dynamics.py` helpers for dense Hamiltonian/unitary construction and
  partial traces.
- Use `shadows.py` helpers for projective and weak partial-shadow feature
  reconstruction.
- Use `diagnostics.py` helpers for projected memory-channel spectra and
  empirical OVD estimates.

Example task run:

```python
result = qres.Experiment(reservoir, dataset, readout=qres.Ridge()).run()
```

For two readout qubits and `pauli_k=2`, the feature count is
`bias + 3*2 + 9 = 16` when `include_bias=True`.

The same core classes are exported from the top-level `pyqres` namespace for
advanced use, for example `pyqres.GlobalFloquetPartialShadowReservoir`.

## Qiskit, MPS, And GPU Simulation

Qiskit-specific code is in:

```text
src/pyqres/qiskit/config.py
src/pyqres/qiskit/reservoir.py
```

`QRCConfig` controls circuit simulation. The dictionary API passes options to
it through the `qiskit` block:

```python
reservoir = qres.qresreservoir.from_dict({
    "memory_qubits": 4,
    "readout_qubits": 2,
    "backend": "qiskit",
    "qiskit": {
        "simulator_method": "statevector",
        "simulator_device": "GPU",
        "use_noise_model": False,
        "aer_options": {"cuStateVec_enable": True},
    },
})
```

Useful modes:

- `statevector` + `GPU`: good for ideal circuit simulation when VRAM is the main
  limit.
- `tensor_network` + `GPU`: useful for some larger structured circuits.
- `matrix_product_state` + `CPU`: useful when entanglement stays local/moderate.
- `density_matrix`: useful for noise, but memory grows quickly.

`QRCReservoir.run_stream(...)` builds a streaming circuit, transpiles it for the
selected Aer backend, executes it, and converts counts into features.

If GPU execution fails, first check:

```python
from qiskit_aer import AerSimulator

print(AerSimulator().available_devices())
print(AerSimulator().available_methods())
```

You should see `GPU` in the devices list.

## Experiments And Data

Generic experiment code is in `src/pyqres/experiments/`.

The most important files are:

- `datasets.py`: `Dataset` and `DatasetSplit`.
- `data.py`: builders for arrays, time series, and `.npz` files.
- `readout.py`: ridge readout.
- `metrics.py`: metric lookup.
- `runner.py`: `Experiment`, `ExperimentResult`, `Sweep`, and `SweepResult`.

`Experiment.run()` does this:

```text
features = qres.run(reservoir, dataset.inputs)
dataset.validate_features(features)
readout.fit(features[train], targets[train])
predictions = readout.predict(...)
metrics = score train/test predictions
return ExperimentResult
```

This is why task packages only need to return a `Dataset`. They do not need to
know anything about quantum reservoirs.

## Task Packages

`pyqres` itself should not contain benchmark tasks. Tasks belong in a separate
package such as `pyqres-tasks`.

A task package should provide things like:

```python
from pyqres_tasks import MackeyGlassConfig, mackey_glass_dataset

dataset = mackey_glass_dataset(MackeyGlassConfig(...))
result = qres.Experiment(reservoir, dataset, metrics=["r2", "mse"]).run()
```

This lets users reuse the same reservoir code for Mackey-Glass, channel
equalization, short-term memory, or their own data.

## Presets

Built-in reservoir presets are in `src/pyqres/presets.py`.

Presets should do one of two things:

- create a generic `ReservoirSpec`
- build lower-level model/Hamiltonian artifacts from a spec

They should not own the experiment loop, dataset generation, or task logic.

When adding a preset:

1. Decide whether it is Hamiltonian-based, circuit-based, or model-based.
2. Add a spec/helper in `presets.py`.
3. Make sure it can fill `ReservoirSpec` fields without requiring a task.
4. Add tests that construct it through `qresreservoir.from_dict(...)`.

## Adding A Custom Reservoir

The lowest-friction extension point is to pass an existing object:

```python
class MyReservoir:
    def run(self, inputs):
        ...
        return features

reservoir = qres.qresreservoir.from_dict({
    "memory_qubits": 1,
    "readout_qubits": 1,
    "dynamics": MyReservoir(),
    "backend": "exact",
})
```

The object does not need to inherit from a pyqres base class. It only needs to
return a finite feature matrix with one row per input sample.

The prethermal shadow reservoir follows the registered preset path:

```python
qres.qresreservoir.from_dict({"preset": "prethermal_shadow", ...})
```

## Adding A Backend

A new backend usually needs changes in three places:

1. Create the backend implementation in a new or existing subpackage.
2. Add a branch in `compile_reservoir(...)`.
3. Add protocol/types/tests if the backend exposes a new public contract.

Keep the backend consuming generic `ReservoirSpec` fields where possible. Avoid
adding task-specific assumptions to `compile_reservoir(...)`.

## Protocols

`core/protocols.py` documents structural contracts. These are not heavy base
classes. They explain what shapes pyqres expects:

- reservoir objects
- dataset objects
- readout objects
- Qiskit config/circuit artifacts
- task package factories

Use protocols when an external package needs to type-check against pyqres
without inheriting from pyqres classes.

## Common Debugging Paths

### The Factory Rejects A Config Field

Error:

```text
ValueError: Unknown reservoir config fields: [...]
```

That means the field is at the wrong level. Common fixes:

- Qiskit/Aer options go under `qiskit`.
- Hamiltonian terms go under `dynamics`.
- Observable settings go under `readout`.
- Model-specific preset parameters go under `dynamics` or `model_kwargs`.

### Feature Shapes Look Wrong

Check:

```python
features = qres.run(reservoir, dataset.inputs)
print(features.shape)
print(dataset.inputs.shape)
print(dataset.targets.shape)
```

There must be one feature row per input sample.

### Qiskit GPU Is Not Used

Check:

```python
from qiskit_aer import AerSimulator
print(AerSimulator().available_devices())
```

If `GPU` is absent, the active Python environment does not have a GPU-enabled
Aer controller loaded.

### Larger Memory-Observable Runs Are Slow

The `memory_observable` backend currently uses dense CPU linear algebra. Larger
qubit counts can become slow because the Hilbert space grows exponentially and
Hamiltonian diagonalization is repeated for input values.

For larger circuit-style experiments, try the Qiskit backend with:

```python
"qiskit": {
    "simulator_method": "statevector",
    "simulator_device": "GPU",
    "use_noise_model": False,
    "aer_options": {"cuStateVec_enable": True},
}
```

or:

```python
"qiskit": {
    "simulator_method": "matrix_product_state",
    "simulator_device": "CPU",
    "use_noise_model": False,
}
```

## Development Checklist

Before finishing a change:

```bash
python -m compileall -q src tests
pytest -q --import-mode=importlib tests
python -m pip check
```

When touching `pyqres-tasks`, run its tests separately to avoid duplicate test
module names:

```bash
pytest -q --import-mode=importlib ../pyqres-tasks/tests
```

## What To Keep Out Of Core

Do not put these into core pyqres:

- Mackey-Glass-specific experiment logic
- channel-equalization-specific experiment logic
- dataset download or acquisition systems
- Hydra/OmegaConf-heavy task runners as the primary API
- assumptions that every reservoir is Ising-like

Core pyqres should remain a library for constructing reservoirs and applying
them to generic supervised data. Presets and tasks should stay as separate
layers on top.
