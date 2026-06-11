"""Qiskit circuit implementation of a streaming quantum reservoir."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from pyqres.core import HamiltonianSpec, ReservoirParams, dense_hamiltonian_matrix

from .config import QRCConfig

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import Operator, SparsePauliOp
    from qiskit.synthesis import LieTrotter, SuzukiTrotter
except Exception:  # pragma: no cover
    QuantumCircuit = None  # type: ignore
    transpile = None  # type: ignore
    PauliEvolutionGate = None  # type: ignore
    Operator = None  # type: ignore
    SparsePauliOp = None  # type: ignore
    LieTrotter = None  # type: ignore
    SuzukiTrotter = None  # type: ignore

try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
except Exception:  # pragma: no cover
    AerSimulator = None  # type: ignore
    NoiseModel = None  # type: ignore

class QRCReservoir:
    """Build and execute streaming reservoir circuits.

    The circuit repeats the same high-level pattern for each scalar input:
    encode input, apply reservoir dynamics, optionally entangle/measure/reset
    ancillas, then measure the system readout. Feature extraction happens from
    the final counts dictionary after Aer execution.
    """

    def __init__(self, cfg: QRCConfig):
        if QuantumCircuit is None:
            raise ImportError("qiskit is required (pip install qiskit qiskit-aer).")
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self._init_backend_parameters()

    def _init_backend_parameters(self) -> None:
        """Initialize reusable Hamiltonian specs and random input signs."""

        cfg = self.cfg
        n = cfg.total_qubits()
        rs = np.random.RandomState(cfg.seed)
        self.input_sign = rs.choice([-1.0, 1.0], size=n)
        if cfg.H0_hamiltonian is None and cfg.H1_hamiltonian is None:
            generated = ReservoirParams.ising_type(
                n_system=cfg.n_system,
                n_ancilla=cfg.n_ancilla,
                tau=cfg.tau,
                seed=cfg.seed,
            ).generate()
            self.H0_hamiltonian = generated["H0_hamiltonian"]
            self.H1_hamiltonian = generated["H1_hamiltonian"]
        else:
            self.H0_hamiltonian = cfg.H0_hamiltonian
            self.H1_hamiltonian = cfg.H1_hamiltonian

    def _to_sparse_pauli_op(self, hamiltonian_like: object | None) -> "SparsePauliOp":
        """Convert backend-neutral Hamiltonian input to Qiskit's sparse Pauli form."""

        n = self.cfg.total_qubits()
        if SparsePauliOp is None or Operator is None:
            raise ImportError("qiskit is required for pauli_evolution reservoirs.")
        if hamiltonian_like is None:
            return SparsePauliOp.from_list([("I" * n, 0.0)])
        if isinstance(hamiltonian_like, HamiltonianSpec):
            return hamiltonian_like.to_sparse_pauli_op()
        if isinstance(hamiltonian_like, SparsePauliOp):
            return hamiltonian_like
        if hasattr(hamiltonian_like, "to_sparse_pauli_op") and callable(hamiltonian_like.to_sparse_pauli_op):
            return hamiltonian_like.to_sparse_pauli_op()
        return SparsePauliOp.from_operator(Operator(dense_hamiltonian_matrix(hamiltonian_like)))

    def _evolution_synthesis(self) -> object | None:
        """Choose a Qiskit product-formula synthesis for Pauli evolution gates."""

        cfg = self.cfg
        if cfg.evolution_synthesis == "default":
            return None
        kwargs = {
            "reps": int(cfg.evolution_reps),
            "insert_barriers": bool(cfg.evolution_insert_barriers),
            "preserve_order": bool(cfg.evolution_preserve_order),
        }
        if cfg.evolution_synthesis == "lie_trotter":
            if LieTrotter is None:
                raise ImportError("qiskit is required for LieTrotter evolution synthesis.")
            return LieTrotter(**kwargs)
        if cfg.evolution_synthesis == "suzuki_trotter":
            if SuzukiTrotter is None:
                raise ImportError("qiskit is required for SuzukiTrotter evolution synthesis.")
            return SuzukiTrotter(order=int(cfg.evolution_order), **kwargs)
        raise ValueError(f"Unsupported evolution_synthesis '{cfg.evolution_synthesis}'.")

    def _apply_pauli_evolution(self, qc: "QuantumCircuit", uval: float) -> None:
        """Append Qiskit-native Pauli evolution for H(u) = H0 + input_scale*u*H1."""

        if PauliEvolutionGate is None:
            raise ImportError("qiskit is required for pauli_evolution reservoirs.")
        h0 = self._to_sparse_pauli_op(self.H0_hamiltonian)
        h1 = self._to_sparse_pauli_op(self.H1_hamiltonian)
        hamiltonian = h0 + float(self.cfg.input_scale * uval) * h1
        qc.append(
            PauliEvolutionGate(hamiltonian, time=float(self.cfg.tau), synthesis=self._evolution_synthesis()),
            list(range(self.cfg.total_qubits())),
        )

    def _apply_encoding(self, qc: "QuantumCircuit", uval: float) -> None:
        """Append the configured scalar-input encoding to an existing circuit."""

        cfg = self.cfg
        n = cfg.total_qubits()
        s = cfg.input_scale
        if cfg.encoding == "rz_global":
            for q in range(n):
                qc.rz(s * uval, q)
        elif cfg.encoding == "rz_per_qubit":
            for q in range(n):
                sgn = 1.0 if cfg.input_map == "global" else float(self.input_sign[q])
                qc.rz(sgn * s * uval, q)
        else:
            raise ValueError(f"Unknown encoding: {cfg.encoding}")

    def _apply_reservoir_unitary(self, qc: "QuantumCircuit") -> None:
        """Append one reservoir-evolution block using the configured ansatz."""

        cfg = self.cfg
        n = cfg.total_qubits()
        if cfg.reservoir_type == "custom_circuit":
            self._apply_custom_circuit(qc)
        elif cfg.reservoir_type == "random_cx_rz":
            for _ in range(cfg.depth_per_step):
                # Shuffle pairings at each layer so the random-CX reservoir does
                # not repeatedly couple the same neighboring indices.
                perm = np.arange(n); self.rng.shuffle(perm)
                pairs = [(perm[k], perm[k+1]) for k in range(0, n-1, 2)]
                for a, b in pairs:
                    qc.cx(a, b); qc.rz(float(self.rng.uniform(-np.pi, np.pi)), b); qc.cx(a, b)
                for q in range(n):
                    qc.rz(float(self.rng.uniform(-np.pi, np.pi)), q)
        else:
            raise ValueError(f"Unknown reservoir_type: {cfg.reservoir_type}")

    def _apply_custom_circuit(self, qc: "QuantumCircuit") -> None:
        """Append a caller-supplied circuit as the fixed reservoir block."""

        cfg = self.cfg
        circuit = cfg.reservoir_circuit
        if circuit is None:
            raise ValueError("reservoir_circuit must be provided when reservoir_type='custom_circuit'.")
        n_circuit = getattr(circuit, "num_qubits", None)
        if n_circuit is None:
            raise TypeError("reservoir_circuit must expose num_qubits.")
        targets = cfg.reservoir_circuit_targets
        if targets is None:
            if int(n_circuit) != cfg.total_qubits():
                raise ValueError(
                    "A custom reservoir circuit without explicit targets must use all "
                    f"{cfg.total_qubits()} qubits; got {n_circuit}."
                )
            targets = tuple(range(cfg.total_qubits()))
        else:
            targets = tuple(int(item) for item in targets)
            if len(targets) != int(n_circuit):
                raise ValueError(
                    f"reservoir_circuit_targets has length {len(targets)}, expected {n_circuit}."
                )
            if len(set(targets)) != len(targets) or any(t < 0 or t >= cfg.total_qubits() for t in targets):
                raise ValueError(f"Invalid reservoir_circuit_targets {targets}.")
        instruction = circuit.to_instruction() if hasattr(circuit, "to_instruction") else circuit
        qc.append(instruction, list(targets))

    def _apply_purification_entangle(self, qc: "QuantumCircuit") -> None:
        """Entangle system and ancilla qubits before ancilla measurement."""

        cfg = self.cfg
        nS, nA = cfg.n_system, cfg.n_ancilla
        if not cfg.use_purification or nA <= 0:
            return
        anc = list(range(nS, nS + nA))
        if cfg.ancilla_pattern == "star":
            for k, aq in enumerate(anc):
                qc.cx(k % nS, aq)
        elif cfg.ancilla_pattern == "pairwise":
            for k, aq in enumerate(anc):
                qc.cx(min(k, nS - 1), aq)
        else:
            raise ValueError(f"Unknown ancilla_pattern: {cfg.ancilla_pattern}")

    def build_streaming_circuit(self, inputs: Sequence[float], measure_system: bool = True) -> Tuple["QuantumCircuit", List[int], List[int]]:
        """Build the full multi-time-step circuit and record bit allocation.

        The returned sys_bits_per_step and anc_bits_per_step arrays are used
        by features_from_counts to decode the flat classical register into one
        feature vector per time step.
        """

        cfg = self.cfg
        nS, nA = cfg.n_system, cfg.n_ancilla
        n = cfg.total_qubits()
        T = len(inputs)
        sys_bits = nS if measure_system else 0
        anc_bits = nA if (cfg.use_purification and nA > 0) else 0
        qc = QuantumCircuit(n, T * (sys_bits + anc_bits))
        cidx = 0
        for uval in inputs:
            if cfg.reservoir_type == "pauli_evolution":
                self._apply_pauli_evolution(qc, float(uval))
            else:
                self._apply_encoding(qc, float(uval))
                self._apply_reservoir_unitary(qc)
            if cfg.use_purification and nA > 0:
                self._apply_purification_entangle(qc)
                for k in range(nA):
                    qc.measure(nS + k, cidx + k)
                cidx += nA
                if cfg.measure_and_reset_ancilla:
                    for k in range(nA):
                        qc.reset(nS + k)
            if measure_system:
                for k in range(nS):
                    qc.measure(k, cidx + k)
                cidx += nS
        return qc, [sys_bits]*T, [anc_bits]*T

    def build_executable_circuit(
        self,
        inputs: Sequence[float],
        backend: Optional[Any] = None,
        measure_system: bool = True,
        optimization_level: Optional[int] = None,
        **transpile_options: Any,
    ) -> "QuantumCircuit":
        """Build a circuit and lower it toward an executable Qiskit backend form.

        Without a backend this returns a decomposed circuit with abstract
        evolution instructions expanded by Qiskit's synthesis plugin. With a
        backend, Qiskit's transpiler maps the circuit to that backend's target
        instruction set and connectivity.
        """

        if transpile is None:
            raise ImportError("qiskit is required to build executable circuits.")
        circuit, _, _ = self.build_streaming_circuit(inputs, measure_system=measure_system)
        if backend is None:
            return circuit.decompose(reps=10)
        level = self.cfg.transpile_optimization_level if optimization_level is None else int(optimization_level)
        return transpile(circuit, backend=backend, optimization_level=level, **transpile_options)

    @staticmethod
    def _bit_at_from_right(bitstring: str, idx: int) -> int:
        return 1 if bitstring[-(idx + 1)] == "1" else 0

    @classmethod
    def _z_expectation_from_counts(cls, counts: Dict[str, int], shots: int, clbit_index: int) -> float:
        """Estimate a single Z expectation value from a counts dictionary."""

        acc = 0.0
        for s, c in counts.items():
            b = cls._bit_at_from_right(s, clbit_index)
            acc += c * (1.0 if b == 0 else -1.0)
        return acc / float(shots)

    @classmethod
    def _z_vector_from_counts(cls, counts: Dict[str, int], shots: int, start: int, n: int) -> np.ndarray:
        z = np.zeros(n, dtype=float)
        for i in range(n):
            z[i] = cls._z_expectation_from_counts(counts, shots, clbit_index=start + i)
        return z

    def features_from_counts(self, counts: Dict[str, int], sys_bits_per_step: List[int], anc_bits_per_step: List[int]) -> np.ndarray:
        """Convert flat Qiskit counts into a dense feature matrix."""

        cfg = self.cfg
        T = len(sys_bits_per_step)
        feats = []
        offset = 0
        for t in range(T):
            a_bits = anc_bits_per_step[t]
            s_bits = sys_bits_per_step[t]
            per = []
            if cfg.readout == "z_local_plus_anc" and a_bits > 0:
                per.extend(self._z_vector_from_counts(counts, cfg.shots, offset, a_bits).tolist())
            offset += a_bits
            if s_bits > 0:
                per.extend(self._z_vector_from_counts(counts, cfg.shots, offset, s_bits).tolist())
            offset += s_bits
            feats.append(np.array(per, dtype=float))
        X = np.vstack(feats)
        if cfg.include_bias:
            X = np.hstack([np.ones((T, 1)), X])
        if not np.isfinite(X).all():
            raise FloatingPointError("Non-finite reservoir features X; reduce shots/noise or check circuit size.")
        return X

    def run_stream(self, inputs: Sequence[float], backend: Optional["AerSimulator"]=None,
                   noise_model: Optional["NoiseModel"]=None, seed_simulator: int = 123) -> np.ndarray:
        """Execute a streaming circuit and return time-indexed readout features."""

        cfg = self.cfg
        qc, sys_bits, anc_bits = self.build_streaming_circuit(inputs, measure_system=True)
        if backend is None:
            if AerSimulator is None:
                raise ImportError("qiskit-aer is required for AerSimulator.")
            if noise_model is None:
                noise_model = cfg.noise.to_noise_model()
            backend = AerSimulator(method=cfg.simulator_method, noise_model=noise_model, seed_simulator=seed_simulator)
        job = backend.run(qc, shots=cfg.shots)
        counts = job.result().get_counts(0)
        return self.features_from_counts(counts, sys_bits, anc_bits)
