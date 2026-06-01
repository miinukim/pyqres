# pyqres

pyqres is the unified quantum reservoir computing package for this workspace.
It brings together the simulation/Qiskit reservoir runtime, benchmark tasks,
classical baselines, and PTM/Volterra dimension analysis that were previously
split across qrclib and qrcdim.

New code should import from pyqres.*. The older package directories can remain
in the workspace during migration, but pyqres now has its own implementation
modules under src/pyqres.

## Package Layout

text
pyqres/
  core/         Shared protocols, measurement control, reservoir parameters
  simulation/   Dense QRC simulation models and simulation frontends
  qiskit/       Qiskit-compatible streaming and noisy reservoirs
  dim/          PTM/Liouville, Volterra, rank, visibility analysis
  tasks/        STM, channel equalization, Mackey-Glass benchmarks
  baselines/    ESN, logistic, and softmax classical baselines
  experiments/  CLI, Hydra configs, sweep helpers, result utilities
  utils/        Internal numerical utilities


The intended dependency direction is inward:

text
pyqres.core
  <- pyqres.simulation
  <- pyqres.qiskit
  <- pyqres.dim
  <- pyqres.tasks
  <- pyqres.baselines
  <- pyqres.experiments


core should stay lightweight. Runtime implementations and analysis code should
depend on it, not the other way around.

## Install

For local development from this directory:

bash
python -m pip install -e .


Install optional groups as needed:

bash
python -m pip install -e ".[qiskit]"
python -m pip install -e ".[dim,experiments]"
python -m pip install -e ".[all]"


The optional groups are organized by use case:

- simulation - dense simulation dependencies
- qiskit - Qiskit and Qiskit Aer execution
- dim - dimension-analysis dependencies
- tasks - benchmark/task dependencies
- experiments - Hydra, pandas, and plotting helpers
- all - full local research environment
- dev - test and lint tooling

## pyqres.core

pyqres.core contains shared abstractions that other subpackages use.

Files:

- core/protocols.py - minimal reservoir protocols and result containers
- core/control.py - measurement, reset, weak-measurement, and feedback helpers
- core/reservoir_params.py - Hamiltonian presets, Pauli-term builders, and coupling generators

Important exports:

python
from pyqres.core import (
    QRCReservoirProtocol,
    ChannelReservoirProtocol,
    CircuitReservoirProtocol,
    ReservoirStepResult,
    ReservoirRunResult,
    HamiltonianSpec,
    MeasurementControlConfig,
    PauliTerm,
    ReservoirParams,
)


Use this layer when adding new reservoir implementations that should be usable
by tasks or analysis code without binding them to one backend.

ReservoirParams supports the built-in Ising-type preset as well as broader
Hamiltonian specifications. Matrix-like inputs can be NumPy arrays, SciPy
sparse matrices, or Qiskit quantum-info operators such as SparsePauliOp and
Operator. Pauli Hamiltonians are carried as backend-neutral HamiltonianSpec
objects so Qiskit/Aer backends can consume SparsePauliOp while the dense
simulation backend only materializes NumPy matrices at its boundary.

python
from pyqres.core import PauliTerm, ReservoirParams
from pyqres.simulation import ExactQRCModel, ExactQRCModelConfig

# Built-in Ising-type preset.
ising_kwargs = ReservoirParams.ising_type(
    n_system=2,
    n_ancilla=1,
).generate()
# With the qiskit extra installed, this stays symbolic for Qiskit/Aer routing.
qiskit_ready_h0 = ising_kwargs["H0_hamiltonian"].to_sparse_pauli_op()
ising_model = ExactQRCModel(ExactQRCModelConfig(**ising_kwargs))

# Arbitrary Pauli-term Hamiltonian H(u) = H0 + u H1.
term_kwargs = ReservoirParams.from_pauli_terms(
    n_system=1,
    n_ancilla=1,
    h0_terms=[PauliTerm(1.0, ((0, "X"),))],
    h1_terms=[PauliTerm(0.5, ((1, "Z"),))],
).generate()
# The dense simulator will convert this at its boundary; Qiskit can use it first.
term_sparse_op = term_kwargs["H0_hamiltonian"].to_sparse_pauli_op()
term_model = ExactQRCModel(ExactQRCModelConfig(**term_kwargs))

# Arbitrary matrix-like Hamiltonians are also accepted.
matrix_kwargs = ReservoirParams.from_matrices(
    n_system=1,
    n_ancilla=1,
    h0_matrix=term_model.H0,
    h1_matrix=term_model.H1,
).generate()
matrix_model = ExactQRCModel(ExactQRCModelConfig(**matrix_kwargs))


## pyqres.simulation

pyqres.simulation contains the dense reservoir simulation implementation and
simulation frontends.

Files:

- simulation/exact_qrc.py - ExactQRCModel and ExactQRCModelConfig
- simulation/channel_map.py - deterministic channel-map reservoir features
- simulation/hardware.py - sampled hardware-trajectory-style reservoir

Important exports:

python
from pyqres.simulation import (
    ExactQRCModel,
    ExactQRCModelConfig,
    ChannelMapReservoir,
    ChannelMapReservoirConfig,
    HardwareTrajectoryReservoir,
    HardwareTrajectoryReservoirConfig,
)


Typical use:

python
import numpy as np

from pyqres.simulation import ChannelMapReservoir, ChannelMapReservoirConfig

cfg = ChannelMapReservoirConfig(
    n_system=2,
    n_ancilla=1,
    tau=0.6,
    input_scale=1.0,
    seed=1,
)
reservoir = ChannelMapReservoir(cfg)
features = reservoir.run(np.linspace(-1.0, 1.0, 20))


Use ExactQRCModel directly when you need channel-level access, dense unitaries,
or PTM-compatible memory-channel operations. The class name still says "Exact"
because it denotes the exact dense simulation backend inside the broader
simulation package.

## pyqres.qiskit

pyqres.qiskit contains the circuit/backend-facing reservoir implementation.

Files:

- qiskit/config.py - Qiskit reservoir and noise configuration dataclasses
- qiskit/reservoir.py - streaming Qiskit reservoir implementations

Important exports:

python
from pyqres.core import ReservoirParams
from pyqres.qiskit import (
    QRCConfig,
    QRCReservoir,
    NISQRCConfig,
    NISQReservoir,
    NoiseConfig,
)

params = ReservoirParams.ising_type(n_system=2, n_ancilla=1).generate()
cfg = QRCConfig(
    n_system=2,
    n_ancilla=1,
    reservoir_type="pauli_evolution",
    H0_hamiltonian=params["H0_hamiltonian"],
    H1_hamiltonian=params["H1_hamiltonian"],
    evolution_synthesis="lie_trotter",
    evolution_reps=1,
)
reservoir = QRCReservoir(cfg)
circuit = reservoir.build_streaming_circuit([0.1], measure_system=False)[0]

# Pass an IBM/Qiskit backend to map the circuit to that backend's target ISA.
backend = None  # replace with service.backend("ibm_backend_name")
executable = reservoir.build_executable_circuit([0.1], backend=backend)


This layer is for circuit-style reservoir execution, Aer simulation, and noisy
NISQ-style experiments. pauli_evolution uses Qiskit's PauliEvolutionGate
with configurable product-formula synthesis. That gate is a high-level
instruction for exp(-iHt); build_executable_circuit decomposes/transpiles it
into basis gates for a concrete backend.

## pyqres.dim

pyqres.dim contains the PTM/Liouville and Volterra-dimension analysis code.

Files:

- dim/pauli.py - Pauli basis and Pauli-string utilities
- dim/linalg_utils.py - PTM coordinates, ranks, null spaces, derivatives
- dim/model.py - Ising, Haar-random, and SYK reservoir models
- dim/analysis.py - affine PTM expansions and Volterra analyzers
- dim/isotropy.py - compressed visibility projector diagnostics
- dim/qrclib_model.py - wrapper from the simulation core into dim analysis
- dim/streaming.py - task-side streaming adapter over the simulation core
- dim/sweep.py - configurable sweep machinery
- dim/experiment_utils.py - reusable experiment table/plot helpers

Important exports:

python
from pyqres.dim import (
    ReservoirBase,
    IsingReservoirModel,
    IsingReservoirParameters,
    QRCLibExactReservoirModel,
    VolterraAnalyzer,
    DenseVolterraAnalyzer,
    TruncatedVolterraGenerator,
    compressed_visibility_diagnostics,
    compressed_visibility_metrics,
)


VolterraAnalyzer is the default analyzer. It constructs the truncated
observable-side Volterra sector directly and avoids building the dense PTM.
DenseVolterraAnalyzer keeps the dense PTM route available for small systems,
debugging, and cross-checks.

Use this layer for questions like:

- What is the latent Volterra span dimension?
- What is the visible Volterra rank?
- Which latent directions are invisible to a chosen readout?
- How isotropic is the readout-visible projector on the latent span?

Example bridge from the simulation runtime into dimension analysis:

python
from pyqres.dim import QRCLibExactReservoirModel
from pyqres.simulation import ExactQRCModelConfig

cfg = ExactQRCModelConfig(n_system=2, n_ancilla=1, seed=1)
model = QRCLibExactReservoirModel(config=cfg)
ptm = model.ptm(0.0)


## pyqres.tasks

pyqres.tasks contains benchmark datasets and task runners.

Files:

- tasks/stm.py - short-term memory benchmark
- tasks/channel_equalization.py - nonlinear channel equalization tasks
- tasks/mackey_glass.py - Mackey-Glass time-series forecasting

Important exports:

python
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


Task runners expect a reservoir object with a reset() method and a run() or
step() interface compatible with the task protocol.

## pyqres.baselines

pyqres.baselines contains classical baselines and readout models used for
comparison.

Files:

- baselines/esn.py - echo-state network implementation and benchmark helpers
- baselines/classical.py - logistic equalizer and multiclass softmax readout

Important exports:

python
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


Use this layer to compare QRC reservoirs against classical recurrent or linear
readout baselines.

## pyqres.experiments

pyqres.experiments contains runnable experiment entry points and sweep helpers.

Files:

- experiments/cli.py - Hydra-driven CLI entry points and benchmark runners
- experiments/conf/ - default Hydra configuration tree
- dim/sweep.py and dim/experiment_utils.py - imported through
  pyqres.experiments for sweep-oriented workflows

Console scripts from pyproject.toml:

bash
pyqres-run
pyqres-stm-demo
pyqres-stm-hydra
pyqres-channel-eq-benchmark


Useful imports:

python
from pyqres.experiments import (
    build_sweep,
    ConfigurableSweep,
    SweepExperiment,
    run_standard_analysis_sweep,
    save_experiment_table,
    save_line_metric_plot,
)


## Workflow Examples

Run a simulated reservoir on a benchmark:

python
from pyqres.simulation import ChannelMapReservoir, ChannelMapReservoirConfig
from pyqres.tasks import STMConfig, STMTaskRunner

reservoir = ChannelMapReservoir(ChannelMapReservoirConfig(n_system=2, n_ancilla=1))
task = STMConfig(T_total=300, washout=50, train_len=150, test_len=75)
scores = STMTaskRunner(reservoir, task).run()


Fit a softmax readout on collected features:

python
from pyqres.baselines import SoftmaxReadoutConfig, fit_softmax_readout, predict_softmax_readout

model = fit_softmax_readout(X_train, y_train, SoftmaxReadoutConfig())
predictions = predict_softmax_readout(X_test, model)


Analyze a reservoir through the default observable-side Volterra tools:

python
from pyqres.dim import QRCLibExactReservoirModel, VolterraAnalyzer
from pyqres.simulation import ExactQRCModelConfig

model = QRCLibExactReservoirModel(
    config=ExactQRCModelConfig(n_system=2, n_ancilla=1, seed=1)
)
analyzer = VolterraAnalyzer(model)


## Development Checks

Run the current smoke tests:

bash
python -m pytest -q


Compile the package:

bash
python -m compileall -q src/pyqres


The smoke tests verify the main public imports and a small simulation/dimension
bridge path.
