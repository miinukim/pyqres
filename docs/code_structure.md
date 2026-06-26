# pyqres Code Structure

This document explains the project at two levels:

- the bigger picture: how the packages fit together and how data flows through an experiment
- the function/class level: what each public and internal function, class, and method is responsible for

It describes the current redesigned package, where `pyqres` core is task-agnostic and named reservoirs are preset adapters rather than assumptions baked into the core.

## Big Picture

`pyqres` is organized around a small core contract:

```text
user config / Python dict
    -> ReservoirSpec + InputEncodingSpec + DynamicsSpec + ReadoutSpec
    -> compile_reservoir(...)
    -> executable reservoir object
    -> transform/run_stream/run(inputs)
    -> feature matrix
    -> Experiment(dataset, readout, metrics).run()
```

The main design separation is:

- `pyqres.core`: generic construction, specs, Hamiltonian wrappers, protocols, and compilation.
- `pyqres.presets`: named adapters such as Ising, RandomPauli, and SYK. Presets fill generic specs or build lower-level backend artifacts.
- `pyqres.simulation`: dense exact simulation backends that consume Hamiltonian artifacts.
- `pyqres.qiskit`: Qiskit circuit backend. Hamiltonians must arrive as Qiskit-native `SparsePauliOp`; circuits arrive as raw `QuantumCircuit`.
- `pyqres.dim`: dimension-analysis and memory-observable reservoirs, including Volterra analysis utilities.
- `pyqres.experiments`: generic dataset/readout/experiment orchestration.
- `pyqres.baselines`: classical ESN and logistic/softmax baselines.
- `pyqres.utils`: small numerical utilities shared by readouts and baselines.
- `qres`: short import alias that re-exports the `pyqres` API.

The intended high-level user path is dictionary-first:

```python
import pyqres as qres

reservoir = qres.qresreservoir.from_dict({
    "preset": "Ising",
    "memory_qubits": 5,
    "readout_qubits": 2,
    "encoding": {"mode": "hamiltonian", "operator": "Z", "targets": [0], "scale": 1.2},
    "dynamics": {"kind": "preset", "name": "Ising", "tau": 0.6},
    "readout": {"mode": "memory_observables", "observables": {"preset": "rich", "count": 8}},
    "backend": "exact",
})

dataset = qres.data.timeseries(series, target_horizon=1).split(
    washout=100,
    train=600,
    test=300,
)

result = qres.Experiment(
    reservoir=reservoir,
    dataset=dataset,
    readout=qres.readout.Ridge(l2=1e-6),
    metrics=["r2", "mse"],
).run()
```

## Construction Flow

1. `qresreservoir.from_dict` accepts plain Python mappings.
2. `qresreservoir.builder_from_dict` normalizes dimensions, seed, encoding, dynamics, readout, backend, and runtime-only objects.
3. The internal builder returned by `builder_from_dict` stores an immutable `ReservoirSpec` plus selected backend.
4. `compile_reservoir` chooses the backend:
   - `exact`/`channel_map`: `ChannelMapReservoir`
   - `hardware`/`hardware_trajectory`: `HardwareTrajectoryReservoir`
   - `memory_observable`/`dim`: `MemoryObservableStreamingReservoir`
   - `qiskit`: `QRCReservoir`
   - `object`: returns the user-supplied reservoir directly
5. `transform` runs any object exposing `transform`, `run_stream`, or `run`.
6. `Experiment.run` fits the readout on train indices and scores train/test metrics.

## Dynamics Model

The core does not require Ising dynamics. It accepts generic dynamics in these forms:

- preset mapping: `{"kind": "preset", "name": "ising", ...}`
- explicit Hamiltonian mapping: `{"kind": "hamiltonian", "parameters": {...}}`
- Hamiltonian aliases directly under `dynamics`: `{"h0_terms": ..., "h1_terms": ...}`
- two-item Hamiltonian pair: `(H0, H1)`
- Qiskit raw circuit object: any object with `num_qubits` and `to_instruction`
- existing reservoir object: any object exposing `transform`, `run_stream`, `run`, or `step`

Preset logic is deliberately separated in `pyqres.presets`. Qiskit-specific conversion is also isolated there: `build_qiskit_artifacts` converts preset Hamiltonian descriptions into `SparsePauliOp` objects before creating `QRCReservoir`.

## Package Exports

`src/pyqres/__init__.py` is the public namespace. It exports:

- construction helpers: `qresreservoir`, `compile_reservoir`, `transform`
- specs: `InputEncodingSpec`, `DynamicsSpec`, `ReadoutSpec`, `ReservoirSpec`
- experiments: `Dataset`, `DatasetSplit`, `Experiment`, `ExperimentResult`, `Sweep`, `SweepResult`, `Ridge`
- protocol types used by users and external packages
- subpackages: `data`, `readout`, `presets`

`src/qres/__init__.py` is a compatibility alias that re-exports `pyqres`.

## Module-Level Inventory

### `pyqres.core.specs`

- `InputEncodingSpec`: immutable description of how user inputs enter the reservoir.
- `InputEncodingSpec.from_mapping`: accepts aliases such as `axis`, `site`, `sites`, and `strength`, normalizes them into `operator`, `targets`, and `scale`, and stores unknown keys under `parameters`.
- `InputEncodingSpec.to_dict`: serializes the encoding spec to JSON/YAML-safe values.
- `DynamicsSpec`: immutable description of reservoir dynamics independent of backend.
- `DynamicsSpec.from_mapping`: normalizes `preset` or `family` aliases into a named preset dynamics spec and moves unknown keys into `parameters`.
- `DynamicsSpec.to_dict`: serializes dynamics to a plain dictionary.
- `ReadoutSpec`: immutable description of how features are extracted from reservoir state.
- `ReadoutSpec.from_mapping`: constructs readout specs and normalizes `custom` observable lists to tuples.
- `ReadoutSpec.to_dict`: serializes readout config, converting tuples/sequences to lists.
- `ReservoirSpec`: complete task-agnostic construction record.
- `ReservoirSpec.with_updates`: returns a copy with selected fields replaced.
- `ReservoirSpec.from_mapping`: builds a full spec from a mapping, recursively normalizing encoding/dynamics/readout.
- `ReservoirSpec.to_dict`: serializes the spec; runtime objects are represented by `repr`.
- `ReservoirSpec.system_qubits`: resolves `n_system` or `n_memory`, raising if neither is set.
- `ReservoirSpec.ancilla_qubits`: resolves `n_ancilla` or `n_readout`, raising if neither is set.

### `pyqres.core.fluent`

- `ReservoirBuilder`: internal inspectable builder over an immutable `ReservoirSpec`.
- `ReservoirBuilder.__init__`: stores a ready reservoir spec and default backend.
- `ReservoirBuilder.spec`: exposes the current spec.
- `ReservoirBuilder.backend`: compiles the current spec with a named backend.
- `ReservoirBuilder.build`: compiles using the selected or supplied backend.
- `_pop_any`: pops the first matching key from a mutable dict.
- `_as_mapping`: validates and copies mapping-like config sections.
- `_looks_like_quantum_circuit`: duck-types raw Qiskit-style circuit dynamics.
- `_looks_like_existing_reservoir`: duck-types already executable reservoir objects.
- `_infer_dynamics`: resolves user dynamics input into a `DynamicsSpec` plus runtime-only objects.
- `_readout_from_config`: normalizes readout mappings into a `ReadoutSpec`.
- `_promoted_spec_updates`: promotes common dynamics parameters such as `tau` and `seed` to spec fields.
- `_spec_from_parts`: assembles a full `ReservoirSpec` from normalized dictionary pieces.
- `qresreservoir`: dictionary-first reservoir factory.
- `qresreservoir.builder_from_dict`: builds a configured `ReservoirBuilder` from a plain mapping.
- `qresreservoir.from_dict`: builds and immediately compiles a reservoir from a mapping.

### `pyqres.core.builders`

- `_select_observable_specs`: resolves observable presets/custom specs/count into a deduplicated list.
- `_preset_key`: delegates preset-name normalization to `pyqres.presets`.
- `build_dimension_model`: builds or returns a dimension-analysis model for memory-observable backends.
- `_normalize_hamiltonian_kwargs`: maps public Hamiltonian aliases such as `H0`, `h0`, and `kind` to `ReservoirParams` field names.
- `build_hamiltonian_params`: returns backend-neutral `H0_hamiltonian` and `H1_hamiltonian` specs from explicit dynamics or presets.
- `compile_reservoir`: central compiler from `ReservoirSpec` to concrete backend reservoir object.
- `transform`: runs a compatible reservoir through `transform`, `run_stream`, or `run`.

### `pyqres.core.reservoir_params`

- `PauliTerm`: normalized symbolic Pauli Hamiltonian term.
- `_kron_all`: dense Kronecker product helper.
- `normalize_pauli_term`: accepts `PauliTerm`, tuple, or mapping and returns `PauliTerm`.
- `pauli_term_matrix`: converts one Pauli term into a dense matrix.
- `pauli_terms_matrix`: sums Pauli term matrices.
- `pauli_terms_to_labels`: converts Pauli terms to Qiskit-compatible `(label, coefficient)` pairs.
- `pauli_terms_to_sparse_pauli_op`: constructs Qiskit's `SparsePauliOp` lazily.
- `dense_hamiltonian_matrix`: converts arrays, scipy sparse matrices, Qiskit operators, or wrapper objects into a dense complex matrix.
- `HamiltonianSpec`: backend-neutral Hamiltonian wrapper.
- `HamiltonianSpec.from_matrix_like`: wraps matrix/operator-like Hamiltonians without densifying.
- `HamiltonianSpec.from_pauli_terms`: wraps symbolic Pauli terms.
- `HamiltonianSpec.to_dense`: materializes a dense matrix.
- `HamiltonianSpec.to_sparse_pauli_op`: materializes a Qiskit `SparsePauliOp`.
- `ReservoirParams`: generator for backend-neutral Hamiltonian parameter dictionaries.
- `ReservoirParams.ising_type`: creates the built-in Ising Hamiltonian generator.
- `ReservoirParams.from_matrices`: creates an explicit matrix Hamiltonian spec.
- `ReservoirParams.from_pauli_terms`: creates an explicit Pauli-term Hamiltonian spec.
- `ReservoirParams.n_qubits`: returns `n_system + n_ancilla`.
- `ReservoirParams._validate_dense_matrix`: checks shape and Hermiticity.
- `ReservoirParams._generate_matrix_hamiltonian`: produces matrix-backed Hamiltonian specs.
- `ReservoirParams._generate_pauli_terms_hamiltonian`: produces Pauli-term-backed specs.
- `ReservoirParams._ising_hamiltonian_specs`: generates symbolic Ising `H0` and `H1`.
- `ReservoirParams.generate`: dispatches matrix, Pauli-term, or Ising generation.

### `pyqres.core.control`

- `MeasurementControlConfig`: configuration for projective/weak measurement, reset/keep behavior, and feedback gates.
- `MeasurementControlConfig.validated`: validates measurement strength and feedback target ranges.
- `single_qubit_gate`: returns dense `X`, `RX`, or `RZ` gate matrices.
- `embed_single_qubit_gate`: embeds a one-qubit gate into a full register.
- `weak_measurement_kraus`: builds weak-measurement Kraus operators for all ancilla outcomes.
- `projective_measurement_kraus`: builds computational-basis projectors for ancilla outcomes.

### `pyqres.core.protocols`

This module defines structural contracts for duck-typed objects. They are documentation plus runtime-checkable guards.

- `ReservoirStepResult`: optional rich single-step feature result.
- `ReservoirRunResult`: optional rich multi-step feature result.
- `TransformReservoirProtocol.transform`: scikit-style feature transform.
- `StreamingReservoirProtocol.run_stream`: stream-oriented feature generation.
- `BatchReservoirProtocol.run`: batch feature generation.
- `StepReservoirProtocol.reset`: reset state before a stream.
- `StepReservoirProtocol.step`: process one scalar input.
- `QRCReservoirProtocol`: exact/trajectory reservoir shape with reset, step, and run.
- `StatefulReservoirProtocol`: stateful reservoir exposing step, stream, and transform.
- `ChannelReservoirProtocol.channel`: apply reduced memory channel.
- `ChannelReservoirProtocol.ptm`: return memory-channel Pauli transfer matrix.
- `CircuitReservoirProtocol.build_streaming_circuit`: build Qiskit circuit and bit layout.
- `CircuitReservoirProtocol.build_executable_circuit`: lower circuit to executable/transpiled form.
- `CircuitReservoirProtocol.features_from_counts`: decode backend counts to features.
- `QuantumCircuitProtocol.to_instruction`: minimal raw circuit contract.
- `SparsePauliOpProtocol`: minimal Qiskit Hamiltonian contract.
- `QiskitReservoirConfigProtocol.total_qubits`: config contract for Qiskit reservoirs.
- `DimensionModelProtocol`: model shape required by memory-observable reservoirs and Volterra analysis.
- `MemoryObservableReservoirProtocol`: streaming readout wrapper over a dimension model.
- `HamiltonianSpecProtocol`: dense and Qiskit Hamiltonian conversion contract.
- `InputEncodingSpecProtocol.to_dict`: serializable encoding contract.
- `DynamicsSpecProtocol.to_dict`: serializable dynamics contract.
- `ReadoutSpecProtocol.to_dict`: serializable readout contract.
- `SerializableSpecProtocol.to_dict`: generic serialization contract.
- `ReservoirSpecProtocol.system_qubits`: resolved memory/system qubit count.
- `ReservoirSpecProtocol.ancilla_qubits`: resolved ancilla/readout qubit count.
- `ReservoirSpecProtocol.with_updates`: immutable update contract.
- `ReservoirBuilderProtocol.spec`: exposes current spec.
- `ReservoirBuilderProtocol.backend`: compile with backend.
- `ReservoirBuilderProtocol.build`: compile current builder.
- `ReservoirCompilerProtocol.__call__`: callable compiler contract.
- `TransformFunctionProtocol.__call__`: contract for generic transform helper.
- `ReservoirFactoryProtocol.builder_from_dict`: dictionary-to-builder contract.
- `ReservoirFactoryProtocol.from_dict`: dictionary-to-reservoir contract.
- `PresetRegistryProtocol`: named preset registry and adapter contract.
- `DynamicsInferenceProtocol.__call__`: dynamics inference contract.
- `DatasetSplitProtocol`: split validation/serialization contract.
- `DatasetProtocol`: dataset validation/persistence contract.
- `SupervisedDataBuilderProtocol.split`: deferred supervised split builder.
- `TimeSeriesDataBuilderProtocol.split`: deferred forecasting split builder.
- `ReadoutProtocol.fit` and `ReadoutProtocol.predict`: supervised readout contract.
- `ExperimentResultProtocol.save`: experiment result persistence contract.
- `ExperimentProtocol.run`: experiment execution contract.
- `SweepResultProtocol.rows` and `save`: sweep result contract.
- `SweepProtocol.specs` and `run`: parameter sweep contract.
- `TaskDatasetFactoryProtocol.__call__`: external task dataset factory contract.
- `TaskRunnerProtocol.run`: external streaming task runner contract.

### `pyqres.presets`

- `ising_memory_readout`: returns a basic Ising memory-observable `ReservoirSpec`.
- `random_pauli_memory_readout`: returns a RandomPauli memory-observable `ReservoirSpec`.
- `syk_memory_readout`: returns an SYK memory-observable `ReservoirSpec`.
- `names`: lists supported preset names.
- `preset_key`: normalizes preset name from a spec or string.
- `build_dimension_model`: builds the correct dimension-analysis model from a preset spec.
- `_dimension_encoding_kwargs`: translates generic encoding metadata to dimension-model constructor kwargs.
- `_hamiltonian_encoding_kwargs`: translates generic Hamiltonian encoding metadata to Ising Hamiltonian generator kwargs.
- `build_hamiltonian_params`: builds backend-neutral Hamiltonians for Hamiltonian-compatible presets.
- `build_qiskit_artifacts`: converts preset Hamiltonian artifacts into Qiskit-native `SparsePauliOp` fields.
- `get`: returns a named preset spec.

### `pyqres.experiments.datasets`

- `DatasetSplit`: washout/train/test index container.
- `DatasetSplit.contiguous`: creates contiguous split ranges from lengths.
- `DatasetSplit.validate`: checks all split indices are one-dimensional and in range.
- `DatasetSplit.to_dict`: serializes split indices.
- `DatasetSplit.from_mapping`: constructs split from explicit index arrays.
- `Dataset`: generic supervised input/target dataset.
- `Dataset.from_arrays`: validates arrays and constructs dataset plus split.
- `Dataset.from_npz`: loads arrays and optional split indices from `.npz`.
- `Dataset.timeseries`: converts scalar series into forecasting inputs/targets.
- `Dataset.validate_features`: checks feature matrix shape, row count, and finiteness.
- `Dataset.save_npz`: persists arrays and split indices.

### `pyqres.experiments.data`

- `SupervisedDataBuilder`: deferred builder for array datasets.
- `SupervisedDataBuilder.__init__`: stores inputs, targets, and metadata.
- `SupervisedDataBuilder.split`: returns `Dataset` from explicit or contiguous split.
- `TimeSeriesDataBuilder`: deferred builder for scalar forecasting datasets.
- `TimeSeriesDataBuilder.__init__`: stores series, horizon, and metadata.
- `TimeSeriesDataBuilder.split`: returns a forecasting `Dataset`.
- `arrays`: starts a supervised array builder.
- `timeseries`: starts a forecasting builder.
- `npz`: loads an `.npz` file into a supervised builder.

### `pyqres.experiments.readout`

- `ReadoutModel`: protocol for readout implementations.
- `Ridge`: linear ridge readout.
- `Ridge._features`: validates feature matrix and optionally prepends bias.
- `Ridge.fit`: fits ridge weights.
- `Ridge.predict`: predicts with fitted weights.

### `pyqres.experiments.metrics`

- `mse`: mean squared error.
- `rmse`: root mean squared error.
- `negative_rmse`: score-style negative RMSE.
- `r2`: coefficient of determination with zero-variance guard.
- `resolve_metrics`: normalizes metric names/callables into a dictionary.

### `pyqres.experiments.runner`

- `ExperimentResult`: result object containing metrics, features, predictions, and metadata.
- `ExperimentResult.save`: writes metrics JSON, metadata JSON, and arrays NPZ.
- `Experiment`: generic supervised experiment runner.
- `Experiment.run`: collects reservoir features, trains readout, predicts train/test/full, and scores metrics.
- `Sweep`: one-parameter sweep over `ReservoirSpec`.
- `Sweep.specs`: materializes all swept specs.
- `Sweep.run`: compiles and runs one experiment per spec.
- `SweepResult`: sweep output container.
- `SweepResult.rows`: flattens sweep metrics into tabular rows.
- `SweepResult.save`: writes sweep CSV and per-run artifacts.
- `_to_builtin`: converts NumPy containers to JSON-safe values.

### `pyqres.experiments.common`

- `to_builtin`: converts NumPy/OmegaConf values to JSON-safe values.
- `dataclass_from_config`: instantiates dataclasses from config sections while rejecting unknown fields.
- `resolve_output_dir`: resolves configured output directory and timestamping.
- `build_model`: builds legacy dimension models from config.
- `select_observable_specs`: resolves observable preset/custom/count config.
- `build_memory_observable_reservoir`: creates a memory-observable streaming reservoir plus spec labels.
- `save_raw_dataset`: stores raw arrays, metadata, and resolved config.
- `_mapping`: converts OmegaConf or mapping to dict.
- `dataset_from_config`: builds `Dataset` from arrays, timeseries, or NPZ config.
- `reservoir_spec_from_config`: builds `ReservoirSpec` from config.
- `readout_from_config`: builds configured readout, currently ridge.
- `run_experiment_from_config`: runs a full generic experiment from a config tree and saves artifacts.

### `pyqres.experiments.cli`

- `run_array_experiment`: convenience function for array experiment metrics.
- `save_features`: writes a feature matrix to `.npy`.
- `_parse_args`: parses CLI arguments.
- `main`: loads YAML config, runs experiment, and prints metrics.

### `pyqres.simulation.exact_qrc`

- `partial_trace_ancilla`: traces out the last subsystem.
- `computational_zero_density`: returns `|0...0><0...0|`.
- `_bits_to_int`: converts bit sequence to integer.
- `_int_to_bits`: converts integer to fixed-length bits.
- `_embed_local_unitary`: embeds a local unitary on arbitrary targets.
- `_statevector_to_preparation_unitary`: builds a unitary that prepares a state from `|0...0>`.
- `_coerce_unitary_matrix`: validates NumPy/Qiskit/unitary-like input as a dense unitary.
- `ExactQRCModelConfig`: dense exact simulation configuration.
- `ExactQRCModel`: dense exact QRC core.
- `ExactQRCModel.__init__`: initializes dimensions, Hamiltonians, measurement operators, and caches.
- `ExactQRCModel.clear_caches`: clears cached unitaries.
- `ExactQRCModel._cache_unitary`: stores a unitary in LRU order.
- `ExactQRCModel._resolve_hamiltonians`: chooses explicit Hamiltonians or generated presets.
- `ExactQRCModel._validate_hamiltonian_matrix`: densifies and validates Hamiltonian matrices.
- `ExactQRCModel._build_measurement_kraus`: builds projective or weak ancilla Kraus operators.
- `ExactQRCModel._encoding_targets_global`: maps local encoding targets to full-register indices.
- `ExactQRCModel._scaled_input`: applies affine scaling to non-Hamiltonian encodings.
- `ExactQRCModel._amplitude_statevector`: builds amplitude-encoding state.
- `ExactQRCModel._input_unitary_local`: builds local amplitude or user-supplied input unitary.
- `ExactQRCModel.encoding_unitary`: embeds input encoding into full register.
- `ExactQRCModel.unitary`: builds or retrieves joint evolution unitary for input `u`.
- `ExactQRCModel.zero_system_state`: returns zero memory density.
- `ExactQRCModel.maximally_mixed_system_state`: returns maximally mixed memory density.
- `ExactQRCModel.initial_system_density`: selects initial memory density.
- `ExactQRCModel.initial_joint_density`: tensors memory state with reset ancilla.
- `ExactQRCModel.evolve_joint`: applies joint unitary.
- `ExactQRCModel._condition_matches`: evaluates feedback condition on measurement outcome.
- `ExactQRCModel.conditioned_gate`: builds outcome-conditioned feedback gate.
- `ExactQRCModel.measurement_branches`: returns unnormalized measurement branches.
- `ExactQRCModel.apply_measurement_protocol_exact`: applies measurement, feedback, normalization, and optional reset exactly.
- `ExactQRCModel.sample_measurement_protocol`: samples one measurement branch for trajectory simulation.
- `ExactQRCModel.system_channel`: applies induced reset memory channel to an operator.
- `ExactQRCModel.exact_step_from_system`: advances reduced memory density and returns ancilla probabilities.

### `pyqres.simulation.channel_map`

- `_kron_all`: dense Kronecker helper.
- `_pauli_basis_matrices`: dense Pauli basis for memory subsystem.
- `ChannelMapReservoirConfig`: exact deterministic reservoir config with feature options.
- `ChannelMapReservoir`: deterministic exact feature reservoir.
- `ChannelMapReservoir.__init__`: initializes exact core, caches, basis, and state.
- `ChannelMapReservoir.reset`: resets memory state.
- `ChannelMapReservoir._memory_channel`: alias for `channel`.
- `ChannelMapReservoir.channel`: applies induced memory channel.
- `ChannelMapReservoir.ptm`: builds Pauli transfer matrix.
- `ChannelMapReservoir.fixed_point`: iterates zero-input channel to stationary density.
- `ChannelMapReservoir.step`: advances one input and emits ancilla probabilities.
- `ChannelMapReservoir.run`: processes a full input stream.
- `ChannelMapReservoir.run_stream`: stream alias.
- `ChannelMapReservoir.transform`: scikit-style alias.

### `pyqres.simulation.hardware`

- `HardwareTrajectoryReservoirConfig`: finite-shot trajectory config.
- `HardwareTrajectoryReservoir`: exact branch-sampling emulator.
- `HardwareTrajectoryReservoir.__init__`: initializes exact core and RNG.
- `HardwareTrajectoryReservoir.run`: samples independent trajectories and returns empirical probabilities.
- `HardwareTrajectoryReservoir.run_stream`: validated stream alias.
- `HardwareTrajectoryReservoir.transform`: scikit-style alias.

### `pyqres.qiskit.config`

- `NoiseConfig`: Qiskit Aer noise-model configuration.
- `NoiseConfig.to_noise_model`: builds Aer damping/depolarizing noise model.
- `QRCConfig`: Qiskit reservoir circuit configuration.
- `QRCConfig.total_qubits`: returns `n_system + n_ancilla`.

### `pyqres.qiskit.reservoir`

- `QRCReservoir`: Qiskit circuit reservoir backend.
- `QRCReservoir.__init__`: validates Qiskit availability and initializes backend artifacts.
- `QRCReservoir._init_backend_parameters`: initializes input signs and required Hamiltonian artifacts.
- `QRCReservoir._require_sparse_pauli_op`: enforces Qiskit-native Hamiltonian inputs.
- `QRCReservoir._evolution_synthesis`: creates product-formula synthesis strategy.
- `QRCReservoir._apply_pauli_evolution`: appends `PauliEvolutionGate` for `H0 + input_scale*u*H1`.
- `QRCReservoir._apply_encoding`: appends scalar `RZ` input encoding.
- `QRCReservoir._apply_reservoir_unitary`: appends random circuit or custom circuit reservoir block.
- `QRCReservoir._apply_custom_circuit`: validates and appends caller-supplied circuit.
- `QRCReservoir._apply_purification_entangle`: entangles system and ancilla before measurement.
- `QRCReservoir.build_streaming_circuit`: builds multi-step circuit and records classical bit layout.
- `QRCReservoir.build_executable_circuit`: decomposes or transpiles circuit for execution.
- `QRCReservoir._bit_at_from_right`: reads a classical bit from a Qiskit count string.
- `QRCReservoir._z_expectation_from_counts`: estimates one `Z` expectation from counts.
- `QRCReservoir._z_vector_from_counts`: estimates multiple `Z` expectations.
- `QRCReservoir.features_from_counts`: converts counts into feature matrix.
- `QRCReservoir.run_stream`: executes circuit on Aer/backend and returns features.

### `pyqres.dim.pauli`

- `kron_all`: cached dense tensor product from Pauli labels.
- `pauli_basis`: cached tensor-product Pauli label basis.
- `pauli_basis_matrices`: cached dense Pauli basis matrices.
- `basis_labels_as_strings`: cached readable Pauli labels.
- `pauli_string`: builds one Pauli string from site labels.
- `single_site_pauli`: one-site Pauli operator.
- `two_site_pauli`: two-site Pauli operator.
- `computational_zero_state`: zero computational basis ket.
- `computational_zero_density`: zero-state density matrix.
- `maximally_mixed`: maximally mixed density matrix.
- `hs_normalization_factor`: Hilbert-Schmidt normalization factor.
- `traceless_basis_indices`: indices excluding all-identity Pauli.

### `pyqres.dim.linalg_utils`

- `NumericalStabilityError`: numerical failure exception.
- `_array_summary`: compact array diagnostic string.
- `ensure_finite`: raises on non-finite arrays.
- `checked_matmul`: matrix multiply with finite checks and warning escalation.
- `ensure_hermiticity`: symmetrizes matrix.
- `partial_trace_last_subsystem`: traces out readout subsystem.
- `operator_to_ptm_coords`: projects operator into PTM coordinates.
- `ptm_coords_to_operator`: reconstructs operator from PTM coordinates.
- `hs_inner_product`: Hilbert-Schmidt inner product.
- `hs_norm`: Hilbert-Schmidt norm.
- `orthogonalize_operator`: Gram-Schmidt step for operator bases.
- `orthonormal_basis_from_columns`: QR-based column-span basis.
- `matrix_rank`: SVD rank with tolerance.
- `null_space`: scipy nullspace wrapper.
- `finite_difference_weights`: finite-difference stencil weights.
- `derivative_from_samples`: applies stencil to array-valued samples.
- `principal_angles`: subspace principal angles.

### `pyqres.dim.model`

- `_sigma_minus`, `_sigma_z`, `_single_qubit_identity`: one-qubit operators for fermionic/SYK construction.
- `_jordan_wigner_annihilation`: dense Jordan-Wigner annihilation operator.
- `_complex_normal_matrix`: complex Gaussian matrix generator.
- `ReservoirBase`: common memory-channel/PTM interface for dimension models.
- `ReservoirBase._initialize_common`: initializes dimensions, Pauli basis, reset state, and caches.
- `ReservoirBase._cache_get`, `_cache_set`, `clear_caches`: cache management.
- `ReservoirBase._memory_site`, `_readout_site`: logical-to-physical site helpers.
- `ReservoirBase._input_physical_sites`: validates and maps input sites.
- `ReservoirBase._input_strength_prefactor`: scales multi-site input strength.
- `ReservoirBase._single`, `_pair`: dense Pauli helper methods.
- `ReservoirBase._memory_edges`, `_memory_next_nearest_edges`: chain edge helpers.
- `ReservoirBase._build_unitary`: abstract subclass hook.
- `ReservoirBase.unitary`: cached unitary builder.
- `ReservoirBase.kraus_operators`: converts joint unitary and reset state into memory Kraus operators.
- `ReservoirBase.channel`: applies memory channel.
- `ReservoirBase.channel_adjoint`: applies adjoint memory channel to observables.
- `ReservoirBase.channel_derivative_adjoint`: finite-difference derivative of adjoint channel.
- `ReservoirBase.ptm`: constructs memory Pauli transfer matrix.
- `ReservoirBase.readout_matrix`: maps observables to traceless PTM coordinates.
- `ReservoirBase.parse_memory_observable`: parses strings like `Z0*X2`.
- `ReservoirBase.default_memory_observable_specs`: returns named observable spec presets.
- `ReservoirBase.default_memory_observables`: materializes observable matrices.
- `ReservoirBase.fixed_point`: iterates zero-input channel to fixed point.
- `IsingReservoirParameters`: dataclass for Ising memory/readout Hamiltonian.
- `IsingReservoirModel`: dense Ising dimension model.
- `IsingReservoirModel.__init__`: initializes common state and precomputes `H0`, `H1`.
- `IsingReservoirModel._build_h0`: assembles static fields and couplings.
- `IsingReservoirModel._build_h1`: assembles input-modulated Pauli drive.
- `IsingReservoirModel.h0`, `h1`: expose dense Hamiltonian components.
- `IsingReservoirModel._build_unitary`: builds `exp(-i tau (H0 + u H1))`.
- `RandomPauliReservoirParameters`: dataclass for random circuit dimension model.
- `RandomPauliReservoirModel`: fixed random circuit with input-dependent reset state.
- `RandomPauliReservoirModel.__init__`: validates params and builds fixed random circuit.
- `RandomPauliReservoirModel._rotation_z`, `_rotation_y`: one-qubit rotations.
- `RandomPauliReservoirModel._random_su2_single_qubit`: random SU(2) gate.
- `RandomPauliReservoirModel._single_site_unitary`: lifts one-qubit gate to full register.
- `RandomPauliReservoirModel._cnot_gate`: dense CNOT in Pauli form.
- `RandomPauliReservoirModel._build_random_circuit`: builds fixed random brickwork circuit.
- `RandomPauliReservoirModel._build_unitary`: returns fixed unitary.
- `RandomPauliReservoirModel._zero_block_state`: zero ket for a block.
- `RandomPauliReservoirModel._ghz_like_state`: GHZ-like input state.
- `RandomPauliReservoirModel._input_reset_state`: input-dependent readout reset density.
- `RandomPauliReservoirModel.kraus_operators`: Kraus operators using input-dependent reset state.
- `SYKReservoirParameters`: dataclass for SYK reservoir.
- `SYKReservoirModel`: number-conserving fermionic SYK dimension model.
- `SYKReservoirModel.__init__`: builds fermionic operators, Hamiltonian, eigendecomposition, and unitary.
- `SYKReservoirModel._build_annihilation`: Jordan-Wigner annihilation wrapper.
- `SYKReservoirModel._build_memory_number_ops`: number operators on memory subsystem.
- `SYKReservoirModel._build_syk4_hamiltonian`: dense SYK4 interaction.
- `SYKReservoirModel._build_syk2_hamiltonian`: dense SYK2 hopping/disorder term.
- `SYKReservoirModel.hamiltonian`, `syk4_hamiltonian`, `syk2_hamiltonian`: expose Hamiltonian copies.
- `SYKReservoirModel._encoded_probability`: maps input to clipped occupation probability.
- `SYKReservoirModel.input_state_vector`: builds one-qubit input state.
- `SYKReservoirModel._zero_block_state`: zero ket for padding.
- `SYKReservoirModel._input_reset_state`: input-dependent readout reset density.
- `SYKReservoirModel._build_unitary`: returns fixed SYK unitary.
- `SYKReservoirModel.kraus_operators`: Kraus operators using input-dependent reset state.
- `SYKReservoirModel.parse_memory_observable`: parses number observables or Pauli observables.
- `SYKReservoirModel.default_memory_observable_specs`: occupation-based observable presets.
- `SYKReservoirModel.particle_number_sector_indices`: basis indices with fixed particle number.
- `SYKReservoirModel.sector_hamiltonian`: Hamiltonian restricted to a particle-number sector.
- `SYKReservoirModel.mean_level_spacing_ratio`: spectral chaos diagnostic.

### `pyqres.dim.streaming`

- `SharedExactStreamingReservoir`: streaming adapter over `ExactQRCModel`.
- `SharedExactStreamingReservoir.__init__`: initializes exact core, readout mode, observables, and state.
- `SharedExactStreamingReservoir.reset`: resets joint state.
- `SharedExactStreamingReservoir.parse_memory_observable`: parses memory observable strings.
- `SharedExactStreamingReservoir._single_site_specs`, `_pair_specs`: observable spec generators.
- `SharedExactStreamingReservoir.default_memory_observable_specs`: named observable presets.
- `SharedExactStreamingReservoir.default_memory_observables`: materializes observables.
- `SharedExactStreamingReservoir._memory_observable_features`: expectation values on reduced memory state.
- `SharedExactStreamingReservoir._ancilla_features`: optional shot-noise ancilla probabilities.
- `SharedExactStreamingReservoir.step`: evolves one input and emits features.
- `SharedExactStreamingReservoir.run_stream`: runs a full stream.
- `SharedExactStreamingReservoir.transform`: scikit-style alias.
- `MemoryObservableStreamingReservoir`: streaming adapter over any dimension model.
- `MemoryObservableStreamingReservoir.__init__`: stores model, observables, bias, and initial state.
- `MemoryObservableStreamingReservoir._initial_density`: builds initial memory density.
- `MemoryObservableStreamingReservoir.reset`: resets memory density.
- `MemoryObservableStreamingReservoir.step`: applies channel and emits observable expectations.
- `MemoryObservableStreamingReservoir.run_stream`: runs full stream.
- `MemoryObservableStreamingReservoir.transform`: scikit-style alias.

### `pyqres.dim.qrclib_model`

- `QRCLibExactReservoirModel`: dimension-analysis wrapper over the dense exact core.
- `QRCLibExactReservoirModel.__init__`: validates reset semantics and initializes common dimension state.
- `QRCLibExactReservoirModel._build_unitary`: delegates to exact core.
- `QRCLibExactReservoirModel.channel`: delegates reduced memory channel to exact core.
- `QRCLibExactReservoirModel.ptm`: builds PTM from exact core channel.
- `QRCLibExactReservoirModel.fixed_point`: iterates zero-input channel.

### `pyqres.dim.analysis`

- `ReservoirModelProtocol`: minimal model interface for Volterra analyzers.
- `VolterraResult`: output diagnostics from dense or observable-side analysis.
- `PTMAffineExpansion`: finite-difference PTM expansion around a reference input.
- `PTMAffineExpansion.__init__`: stores settings and computes derivatives.
- `PTMAffineExpansion._compute`: samples PTMs, finds fixed point, and computes derivative blocks.
- `PTMAffineExpansion.A0`: zero-order traceless PTM block.
- `PTMAffineExpansion.Ak`: kth derivative of linear block.
- `PTMAffineExpansion.bk`: kth derivative of affine offset.
- `TruncatedVolterraGenerator`: generates finite polynomial-history Volterra family.
- `TruncatedVolterraGenerator.__init__`: stores expansion and truncation.
- `TruncatedVolterraGenerator.generate`: returns monomial labels and latent kernel matrix.
- `_ObservableWordState`: internal stack state for observable-side basis enumeration.
- `_noise_threshold`: statistical visibility threshold.
- `_empty_result`: shaped empty `VolterraResult`.
- `_ambient_readout_matrix`: flattened observable readout matrix.
- `_finalize_result`: computes singular spectra, VVR, OVD, angles, and final result.
- `DenseVolterraAnalyzer`: dense PTM validation analyzer.
- `DenseVolterraAnalyzer.__init__`: builds PTM expansion and generator.
- `DenseVolterraAnalyzer.analyze`: computes dense latent/visible diagnostics.
- `ObservableVolterraBasisBuilder`: matrix-free observable-side basis generator.
- `ObservableVolterraBasisBuilder.__init__`: stores model, seeds, truncation, finite-difference settings.
- `ObservableVolterraBasisBuilder.build`: enumerates drift/insertion words and orthonormalizes operators.
- `VolterraAnalyzer`: main observable-side analyzer.
- `VolterraAnalyzer.__init__`: builds observable-side basis builder.
- `VolterraAnalyzer._restricted_measurement_matrix`: computes readout overlaps on basis operators.
- `VolterraAnalyzer.analyze`: builds basis, computes restricted matrix, and finalizes diagnostics.

### `pyqres.dim.isotropy`

- `_encode_complex_matrix`: JSON-encodes complex matrix real/imag parts.
- `_op_norm_hermitian`: Hermitian operator norm.
- `_max_offdiag_row_sum`: off-diagonal leakage diagnostic.
- `_orth_projector`: orthogonal projector onto column span.
- `_visibility_angles_deg`: converts `sin^2(theta)` values to degrees.
- `_theta_star_deg`: scalar visibility angle from alpha.
- `CompressedVisibilityDiagnostics`: structured visibility-projector diagnostics.
- `CompressedVisibilityDiagnostics.s_gamma_plus`: supported latent dimension.
- `CompressedVisibilityDiagnostics.exact_null_fraction`: fraction of latent directions invisible to readout.
- `CompressedVisibilityDiagnostics.zero_component_fraction`: alias for exact-null fraction.
- `CompressedVisibilityDiagnostics.as_metrics_dict`: flattens diagnostics for CSV/JSON.
- `compressed_visibility_diagnostics`: computes compressed visibility projectors and isotropy metrics.
- `compressed_visibility_metrics`: returns flat metrics dictionary.

### `pyqres.dim.sweep`

- `SweepFamilyProtocol`: sweep family interface.
- `SweepRule`: one field update rule.
- `SweepRule.apply`: applies replace/add/multiply rule to a sweep value.
- `ConfigurableSweep`: sweep family assembled from config.
- `ConfigurableSweep.parameter_label`: returns plot/table label.
- `ConfigurableSweep.parameters`: materializes parameter dataclass for a sweep point.
- `ConfigurableSweep.build_model`: builds model from parameters.
- `_normalize_mapping`: plain dict conversion.
- `_sweep_rules_from_config`: builds `SweepRule` objects from config.
- `build_sweep`: constructs a configurable sweep from mapping config.
- `SweepExperiment`: runs Volterra analysis across sweep values.
- `SweepExperiment.__init__`: stores sweep and analysis settings.
- `SweepExperiment.run`: returns a dataframe of Volterra diagnostics.
- `SweepExperiment.save`: writes CSV and summary plots.

### `pyqres.dim.experiment_utils`

- `LineMetricSpec`: plotting style for one metric line.
- `sweep_values_from_cfg`: creates a linspace from config.
- `_cfg_float`, `_cfg_int`: config value helpers.
- `_pauli_spec_from_word`: converts Pauli word to compact observable spec.
- `_random_pauli_observable_specs`: samples random Pauli observable specs.
- `_observable_specs_from_cfg`: combines preset, custom, and random observables.
- `_standard_analysis_row`: flattens successful `VolterraResult`.
- `_standard_failure_row`: flattens failed sweep point.
- `run_standard_analysis_sweep`: common build-model/analyze/store loop for dimension experiments.
- `save_experiment_table`: writes CSV and resolved config.
- `save_line_metric_plot`: saves selected dataframe metrics as a line plot.

### `pyqres.baselines.classical`

- `LogisticEqualizerConfig`: binary logistic equalizer settings.
- `SoftmaxReadoutConfig`: multiclass softmax readout settings.
- `_sigmoid`: stable sigmoid.
- `_lagged_design_matrix`: builds lagged scalar observation features.
- `_message_lagged_design`: builds lagged features for batched messages.
- `_prepare_features`: validates 2D features and optionally adds intercept.
- `fit_softmax_readout`: trains multiclass regularized linear readout with L-BFGS.
- `predict_softmax_readout`: predicts class labels from fitted softmax model.
- `_fit_logistic_regression`: trains binary logistic regression with Newton iterations.
- `run_channel_equalization_logistic`: binary logistic channel-equalization baseline.
- `run_channel_equalization_symbol_logistic`: multiclass symbol-level equalizer.

### `pyqres.baselines.esn`

- `ESNConfig`: ESN baseline hyperparameters.
- `EchoStateNetwork`: leaky echo-state network.
- `EchoStateNetwork.__init__`: initializes recurrent/input weights and state.
- `EchoStateNetwork.step`: advances one scalar input.
- `EchoStateNetwork.collect_states`: returns bias-augmented state matrix.
- `run_stm_esn`: short-term memory benchmark baseline.
- `run_channel_equalization_esn`: binary channel equalization baseline.

### `pyqres.utils.linear`

- `ridge_regression_fit`: solves ridge normal equations with unregularized first column.
- `ridge_regression_predict`: applies fitted ridge weights.
- `rmse`: root mean squared error.
- `r2_score`: coefficient of determination.

## Notes On Current Design Boundaries

- The core is generic by spec and protocol. The old chainable builder surface has been removed from public exports; use `qresreservoir.from_dict`.
- The Qiskit backend intentionally rejects pyqres Hamiltonian wrappers at the low level. Presets must be transformed to `SparsePauliOp` by `presets.build_qiskit_artifacts`, or users must supply Qiskit-native Hamiltonians themselves.
- `pyqres.experiments.common` still contains compatibility helpers for config-driven workflows. The lighter task-agnostic path is `qresreservoir.from_dict` plus `Experiment`.
- Task presets should live outside this package, for example in `pyqres-tasks`, and should produce generic `Dataset` objects or call the generic reservoir transform API.
