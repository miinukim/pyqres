# pyqres

pyqres is a unified quantum reservoir computing library. It consolidates the
runtime ideas from qrclib and the PTM, Volterra, and dimensional-analysis tools
from qrcdim into one package with a shared set of abstractions.

The package is organized around one central modeling convention:

    H(u) = H0 + input_scale * u * H1

H0 is the fixed reservoir Hamiltonian. H1 is the input-modulated Hamiltonian.
Both are represented at the public API boundary as backend-neutral Hamiltonian
objects when possible. Dense simulation can materialize them as NumPy matrices,
while Qiskit can consume them as SparsePauliOp objects through PauliEvolutionGate.

The goal is not to make the library Ising-specific. The Ising reservoir is only
one preset for generating H0 and H1. Users can also provide arbitrary dense
matrices, SciPy sparse matrices, Qiskit operator objects, or Pauli-term lists.

## Repository Layout

    pyqres/
      pyproject.toml
      README.md
      src/pyqres/
        core/
        simulation/
        qiskit/
        dim/
        tasks/
        baselines/
        experiments/
        utils/
      tests/

The package under src/pyqres is the implementation. The top-level experiments
directory in the workspace is for project experiments that use pyqres as a
library; it is intentionally separate from the installable package.

## Package Dependency Direction

The modules are intended to depend inward:

    core
      used by simulation
      used by qiskit
      used by dim
      used by tasks
      used by baselines
      used by experiments

core should remain lightweight and backend-neutral. Simulation, Qiskit,
analysis, tasks, and experiment code should import common configuration and
protocol objects from core rather than inventing their own incompatible types.

## Installation

For local development:

    python -m pip install -e .

Install optional feature groups as needed:

    python -m pip install -e .[simulation]
    python -m pip install -e .[qiskit]
    python -m pip install -e .[dim]
    python -m pip install -e .[tasks]
    python -m pip install -e .[experiments]
    python -m pip install -e .[all]
    python -m pip install -e .[dev]

Optional groups:

- simulation: SciPy support for dense simulation.
- qiskit: Qiskit and Qiskit Aer execution.
- dim: SciPy support for PTM and Volterra analysis.
- tasks: benchmark task dependencies.
- experiments: Hydra, pandas, and matplotlib.
- all: full research environment.
- dev: pytest and lint tooling.

## High-Level Workflow

A typical pyqres workflow has four stages:

1. Define reservoir Hamiltonians.

       from pyqres.core import PauliTerm, ReservoirParams

       params = ReservoirParams.from_pauli_terms(
           n_system=2,
           n_ancilla=1,
           h0_terms=[
               PauliTerm(1.0, ((0, "X"),)),
               PauliTerm(0.7, ((0, "Z"), (1, "Z"))),
           ],
           h1_terms=[
               PauliTerm(0.5, ((2, "Z"),)),
           ],
       ).generate()

2. Build a reservoir runtime.

       from pyqres.simulation import ChannelMapReservoir, ChannelMapReservoirConfig

       reservoir = ChannelMapReservoir(
           ChannelMapReservoirConfig(
               n_system=2,
               n_ancilla=1,
               H0_hamiltonian=params["H0_hamiltonian"],
               H1_hamiltonian=params["H1_hamiltonian"],
               tau=0.8,
               input_scale=1.0,
               seed=1,
           )
       )

3. Run a task or collect features.

       import numpy as np

       inputs = np.linspace(-1.0, 1.0, 100)
       features = reservoir.run_stream(inputs.tolist())

4. Analyze the same model through PTM or Volterra tools.

       from pyqres.dim import QRCLibExactReservoirModel, VolterraAnalyzer
       from pyqres.simulation import ExactQRCModelConfig

       model = QRCLibExactReservoirModel(
           config=ExactQRCModelConfig(
               n_system=2,
               n_ancilla=1,
               H0_hamiltonian=params["H0_hamiltonian"],
               H1_hamiltonian=params["H1_hamiltonian"],
               seed=1,
           )
       )
       analyzer = VolterraAnalyzer(model)

## Core Module

Package: src/pyqres/core

Purpose:

core defines shared, backend-neutral objects. It is the contract layer used by
simulation, Qiskit, tasks, and dimensional analysis.

Files and capabilities:

- core/protocols.py
  - ReservoirStepResult: immutable per-step result container.
  - ReservoirRunResult: immutable full-stream result container.
  - QRCReservoirProtocol: minimal reservoir interface.
  - ChannelReservoirProtocol: protocol for channel-style reservoirs.
  - CircuitReservoirProtocol: protocol for circuit-style reservoirs.

- core/control.py
  - MeasurementControlConfig: describes ancilla measurement, weak measurement,
    post-measurement reset behavior, and optional conditioned feedback gates.
  - single_qubit_gate: builds one-qubit dense gates.
  - embed_single_qubit_gate: embeds a one-qubit gate into a larger register.
  - weak_measurement_kraus: creates weak-measurement Kraus operators.
  - projective_measurement_kraus: creates computational-basis projective Kraus
    operators for the ancilla register.

- core/reservoir_params.py
  - PauliTerm: one symbolic Pauli Hamiltonian term with a coefficient and site
    operators.
  - HamiltonianSpec: backend-neutral Hamiltonian wrapper. It can materialize as
    a dense matrix or, when Qiskit is installed, as SparsePauliOp.
  - ReservoirParams: generator and wrapper for H0 and H1 Hamiltonian
    specifications.
  - normalize_pauli_term: normalizes tuple, dict, or PauliTerm inputs.
  - pauli_term_matrix and pauli_terms_matrix: dense NumPy construction from
    Pauli terms.
  - pauli_terms_to_labels: converts terms to Pauli-string labels.
  - pauli_terms_to_sparse_pauli_op: converts terms to Qiskit SparsePauliOp.
  - dense_hamiltonian_matrix: accepts common matrix-like objects and returns a
    dense complex NumPy matrix.

ReservoirParams construction modes:

- ReservoirParams.ising_type: built-in open-boundary nearest-neighbor Ising
  preset. It returns H0_hamiltonian and H1_hamiltonian; it is not the central
  representation of the rest of the package.
- ReservoirParams.from_pauli_terms: arbitrary symbolic Pauli Hamiltonians.
- ReservoirParams.from_matrices: arbitrary matrix-like H0 and H1.

Example:

    from pyqres.core import PauliTerm, ReservoirParams

    params = ReservoirParams.from_pauli_terms(
        n_system=1,
        n_ancilla=1,
        h0_terms=[PauliTerm(1.0, ((0, "X"),))],
        h1_terms=[PauliTerm(0.5, ((1, "Z"),))],
    ).generate()

    H0 = params["H0_hamiltonian"]
    H1 = params["H1_hamiltonian"]

## Simulation Module

Package: src/pyqres/simulation

Purpose:

simulation contains dense, exact numerical reservoir runtimes. These classes
are useful for small systems, deterministic feature extraction, debugging, and
as the backend for dimension-analysis adapters.

Files and capabilities:

- simulation/exact_qrc.py
  - ExactQRCModelConfig: configuration for dense simulation. It accepts H0 and
    H1 as HamiltonianSpec, matrices, sparse matrices, or Qiskit-like operators.
    It also configures input encoding, input scaling, unitary caching,
    measurement control, and ancilla reset behavior.
  - ExactQRCModel: dense QRC core. It builds U(u), evolves joint density
    matrices, applies Kraus measurement protocols, samples measurement
    trajectories, returns reduced memory channels, computes fixed points, and
    exposes channel operations needed by PTM analysis.
  - partial_trace_ancilla: traces out the ancilla register.
  - computational_zero_density: builds a zero-state density matrix.

- simulation/channel_map.py
  - ChannelMapReservoirConfig: feature-extraction configuration extending
    ExactQRCModelConfig.
  - ChannelMapReservoir: deterministic reduced-state reservoir. It tracks only
    the memory density matrix between steps, resets ancilla internally, and
    returns expectation-value style features. It can optionally add finite-shot
    sampling noise to features.

- simulation/hardware.py
  - HardwareTrajectoryReservoirConfig: finite-shot trajectory configuration.
  - HardwareTrajectoryReservoir: samples measurement outcomes shot by shot
    using the dense exact core. This is still a simulator, but its output is
    closer to hardware count statistics than ChannelMapReservoir.

Typical deterministic simulation:

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
    features = reservoir.run_stream(np.linspace(-1.0, 1.0, 20).tolist())

When to use which class:

- Use ExactQRCModel when you need dense channel access, unitaries, fixed points,
  PTM-compatible operations, or custom analysis.
- Use ChannelMapReservoir when you need deterministic feature matrices for
  tasks and sweeps.
- Use HardwareTrajectoryReservoir when finite-shot measurement branching is
  important.

## Qiskit Module

Package: src/pyqres/qiskit

Purpose:

qiskit provides circuit-level reservoir construction and execution. It is the
path for Aer simulation and eventual backend transpilation.

Files and capabilities:

- qiskit/config.py
  - NoiseConfig: builds Qiskit Aer noise models with damping and depolarizing
    components.
  - QRCConfig: circuit reservoir configuration. It controls qubit counts,
    reservoir type, H0/H1 Hamiltonians, evolution synthesis, depth, time step,
    input scaling, measurement, readout, shots, simulator method, and transpile
    options.

- qiskit/reservoir.py
  - QRCReservoir: builds streaming QuantumCircuit objects, appends Qiskit-native
    PauliEvolutionGate instructions for H(u), decomposes or transpiles circuits
    into executable backend forms, runs Aer simulations, and converts counts
    into time-indexed feature matrices.

Supported reservoir types:

- pauli_evolution: the default. It uses H0_hamiltonian and H1_hamiltonian to
  build H(u), then appends PauliEvolutionGate.
- random_cx_rz: a random circuit ansatz with CX and RZ layers. This is useful
  as a circuit ansatz baseline, not as a Hamiltonian simulation.

Qiskit example:

    from pyqres.core import ReservoirParams
    from pyqres.qiskit import QRCConfig, QRCReservoir

    params = ReservoirParams.ising_type(n_system=2, n_ancilla=1).generate()
    cfg = QRCConfig(
        n_system=2,
        n_ancilla=1,
        H0_hamiltonian=params["H0_hamiltonian"],
        H1_hamiltonian=params["H1_hamiltonian"],
        reservoir_type="pauli_evolution",
        evolution_synthesis="lie_trotter",
        evolution_reps=1,
    )
    reservoir = QRCReservoir(cfg)
    circuit = reservoir.build_streaming_circuit([0.1, 0.2], measure_system=False)[0]
    executable = reservoir.build_executable_circuit([0.1, 0.2], backend=None)

Unitary time evolution in Qiskit:

PauliEvolutionGate represents exp(-i H t) at the circuit level. It is a
high-level instruction, not a native hardware gate. build_executable_circuit
decomposes or transpiles it through Qiskit's product-formula synthesis and
backend transpiler so it becomes a sequence of basis gates for execution.

## Dimension Analysis Module

Package: src/pyqres/dim

Purpose:

dim contains PTM, Liouville, Volterra, visibility, and sweep-analysis tools.
It is the successor to the qrcdim-style analysis workflow.

Files and capabilities:

- dim/pauli.py
  - Pauli basis construction, dense Pauli-string matrices, computational zero
    states, maximally mixed states, Hilbert-Schmidt normalization factors, and
    traceless-basis indexing.

- dim/linalg_utils.py
  - NumericalStabilityError: error type for numerical validation failures.
  - ensure_finite, checked_matmul, ensure_hermiticity: numerical safety helpers.
  - operator_to_ptm_coords and ptm_coords_to_operator: maps between operators
    and PTM coordinates.
  - hs_inner_product and hs_norm: Hilbert-Schmidt geometry.
  - orthogonalize_operator, orthonormal_basis_from_columns, matrix_rank,
    null_space: subspace utilities.
  - finite_difference_weights and derivative_from_samples: finite-difference
    derivative estimation.
  - principal_angles: subspace-angle computation used internally. Public
    visibility angles are reported in degrees.

- dim/model.py
  - ReservoirBase: abstract base for reservoir models used by analysis.
  - IsingReservoirParameters and IsingReservoirModel: open-boundary Ising
    analysis model.
  - RandomPauliReservoirParameters and RandomPauliReservoirModel: random Pauli
    circuit-style reservoir model.
  - SYKReservoirParameters and SYKReservoirModel: SYK-style fermionic reservoir
    model.

- dim/qrclib_model.py
  - QRCLibExactReservoirModel: adapter that wraps simulation.ExactQRCModel so
    the same exact simulator can be analyzed by the Volterra and PTM tools.

- dim/analysis.py
  - ReservoirModelProtocol: interface expected by Volterra analysis.
  - VolterraResult: result container for VVR, OVD, latent dimension, singular
    values, and visibility angles.
  - PTMAffineExpansion: dense PTM finite-difference expansion around an input
    point.
  - TruncatedVolterraGenerator: dense polynomial-history Volterra generator.
  - DenseVolterraAnalyzer: dense PTM-based analyzer for small-system validation.
  - ObservableVolterraBasisBuilder: constructs the observable-side Volterra
    sector directly.
  - VolterraAnalyzer: the main analyzer. It avoids building dense PTMs and
    works on the observable-side construction.

- dim/isotropy.py
  - CompressedVisibilityDiagnostics: structured visibility and isotropy result.
  - compressed_visibility_diagnostics: detailed projector and visibility-angle
    diagnostics.
  - compressed_visibility_metrics: compact metrics dictionary for experiments.

- dim/streaming.py
  - SharedExactStreamingReservoir: streaming adapter over the exact simulator.
  - MemoryObservableStreamingReservoir: streams chosen memory observables as
    task features.

- dim/sweep.py
  - SweepRule: one parameter sweep rule.
  - ConfigurableSweep: applies sweep rules to nested config dictionaries.
  - build_sweep: builds a sweep object from config.
  - SweepExperiment: evaluates a sweep family and collects rows.

- dim/experiment_utils.py
  - LineMetricSpec: plotting metric specification.
  - sweep_values_from_cfg: converts Hydra-style sweep config to values.
  - run_standard_analysis_sweep: common VVR, OVD, and visibility sweep runner.
  - save_experiment_table: writes tabular results.
  - save_line_metric_plot: writes line plots for selected metrics.

Main analysis objects:

- VolterraAnalyzer: default analyzer. It constructs the observable-side
  Volterra sector directly and is the preferred path for VVR, OVD, and
  visibility-angle diagnostics.
- DenseVolterraAnalyzer: dense PTM route. Use it for small systems, debugging,
  and cross-checking.
- QRCLibExactReservoirModel: bridge from the simulation runtime to analysis.

Example:

    from pyqres.dim import QRCLibExactReservoirModel, VolterraAnalyzer
    from pyqres.simulation import ExactQRCModelConfig

    model = QRCLibExactReservoirModel(
        config=ExactQRCModelConfig(n_system=2, n_ancilla=1, seed=1)
    )
    observables = model.default_memory_observables(preset="z")
    analyzer = VolterraAnalyzer(
        model,
        observables=observables,
        max_order=2,
        lag_horizon=2,
    )
    result = analyzer.analyze()

Important metrics:

- VVR: visible Volterra rank. It is the rank of the visible coefficient matrix.
- OVD: observable Volterra dimension. It counts visible directions that survive
  the configured numerical threshold.
- latent_dim: dimension of the generated latent Volterra sector.
- principal_angles_deg: visibility angles in degrees between latent and
  readout-visible subspaces.

## Tasks Module

Package: src/pyqres/tasks

Purpose:

tasks contains backend-neutral benchmark datasets and task runners. A task
runner only expects a reservoir with the small streaming interface, so exact
simulation, Qiskit-backed reservoirs, and future reservoirs can be swapped in.

Files and capabilities:

- tasks/stm.py
  - STMConfig: stream length, train/test split, delays, input distribution,
    ridge regularization, and metric.
  - STMTaskRunner: generates scalar inputs, collects reservoir features, fits
    one ridge readout per delay, and reports train/test recall scores.
  - memory_capacity: sums positive delay scores as a memory-capacity proxy.

- tasks/mackey_glass.py
  - MackeyGlassConfig: time-series generation, prediction horizon, split, and
    scoring settings.
  - generate_mackey_glass_series: Euler integration of the Mackey-Glass system.
  - MackeyGlassTaskRunner: forecasts future values from reservoir features.

- tasks/channel_equalization.py
  - ChannelEqualizationConfig: classic continuous binary channel equalization.
  - ChannelEqualizationDatasetConfig: symbol-level train/test dataset
    generation.
  - generate_channel_equalization_data: continuous channel-eq input/target
    generation.
  - generate_channel_equalization_dataset: independent symbol-message dataset.
  - ChannelEqualizationTaskRunner: collects features and fits a linear
    equalizer.
  - collect_channel_equalization_reservoir_features: resets a reservoir for
    each independent observed message and stacks features.

Task example:

    from pyqres.simulation import ChannelMapReservoir, ChannelMapReservoirConfig
    from pyqres.tasks import STMConfig, STMTaskRunner

    reservoir = ChannelMapReservoir(
        ChannelMapReservoirConfig(n_system=2, n_ancilla=1, seed=1)
    )
    task = STMConfig(T_total=300, washout=50, train_len=150, test_len=75)
    scores = STMTaskRunner(reservoir, task).run()

## Baselines Module

Package: src/pyqres/baselines

Purpose:

baselines contains classical comparison models and readout helpers used by the
task and experiment layers.

Files and capabilities:

- baselines/esn.py
  - ESNConfig: echo-state network size, spectral radius, input scale, leak rate,
    ridge penalty, seed, clipping, and spectral-radius iteration settings.
  - EchoStateNetwork: sparse recurrent reservoir with leaky state updates,
    feature collection, and ridge readout fitting helpers.
  - run_stm_esn: ESN baseline for STM.
  - run_channel_equalization_esn: ESN baseline for channel equalization.

- baselines/classical.py
  - LogisticEqualizerConfig: binary logistic equalizer settings.
  - SoftmaxReadoutConfig: multiclass softmax readout settings.
  - fit_softmax_readout and predict_softmax_readout: multiclass readout.
  - run_channel_equalization_logistic: continuous channel-eq logistic baseline.
  - run_channel_equalization_symbol_logistic: symbol-level logistic baseline.

Softmax example:

    from pyqres.baselines import SoftmaxReadoutConfig
    from pyqres.baselines import fit_softmax_readout, predict_softmax_readout

    model = fit_softmax_readout(X_train, y_train, SoftmaxReadoutConfig())
    predictions = predict_softmax_readout(X_test, model)

## Experiments Module

Package: src/pyqres/experiments

Purpose:

experiments contains installable CLI entry points, Hydra configs, and reusable
sweep helpers. Workspace-level experiment scripts can import this package but
should live outside the library when they are specific to one study.

Files and capabilities:

- experiments/cli.py
  - pyqres-run: general Hydra experiment runner.
  - pyqres-stm-demo: quick STM demo.
  - pyqres-stm-hydra: Hydra-configured STM runner.
  - pyqres-channel-eq-benchmark: symbol-level channel-eq benchmark.
  - run_experiment_from_cfg: programmatic config runner.
  - run_symbol_channel_equalization_benchmark_from_cfg: programmatic SNR sweep.

- experiments/common.py
  - dataclass_from_config: maps config sections into dataclass instances.
  - build_model and build_task_config: build supported model and task objects.
  - build_memory_observable_reservoir: selects observables and builds the
    memory-observable task reservoir.
  - resolve_output_dir and save_raw_dataset: handle raw dataset output without
    plotting or sweep logic.

- experiments/conf/config.yaml
  - Default config for the general runner.

- experiments/conf/channel_equalization_benchmark.yaml
  - Default config for the symbol-level channel equalization benchmark.

The experiments package also re-exports sweep helpers from pyqres.dim:

- ConfigurableSweep
- SweepExperiment
- build_sweep
- run_standard_analysis_sweep
- save_experiment_table
- save_line_metric_plot
- sweep_values_from_cfg

CLI examples:

    pyqres-stm-demo

    pyqres-run task=stm reservoir.kind=channel_map

    pyqres-channel-eq-benchmark reservoir_params.n_system=2 reservoir_params.n_ancilla=4

## Utils Module

Package: src/pyqres/utils

Purpose:

utils contains small numerical helpers that are shared by tasks and baselines.

Files and capabilities:

- utils/linear.py
  - ridge_regression_fit: closed-form ridge readout fitting.
  - ridge_regression_predict: dense linear prediction.
  - rmse: root mean squared error.
  - r2_score: coefficient of determination.

## Public Namespace Summary

Top-level package:

- pyqres exports the core protocols and result containers.

Use specific subpackages for actual functionality:

- pyqres.core for Hamiltonian specifications, Pauli terms, measurement control,
  and protocols.
- pyqres.simulation for dense exact runtime and channel-map features.
- pyqres.qiskit for circuit construction, Aer execution, and transpilation.
- pyqres.dim for PTM, Volterra, visibility, and sweep analysis.
- pyqres.tasks for benchmark datasets and task runners.
- pyqres.baselines for ESN, logistic, and softmax baselines.
- pyqres.experiments for CLI entry points and reusable experiment utilities.
- pyqres.utils for shared numerical helpers.

## Development Checks

Run tests:

    python -m pytest -q

Compile the package:

    python -m compileall -q src/pyqres tests

Scan for literal backtick characters:

    rg -n \x60 src tests README.md

The smoke tests currently verify public imports, HamiltonianSpec construction,
dense simulation, Qiskit-compatible Hamiltonian inputs, and the simulation to
dimension-analysis bridge.
