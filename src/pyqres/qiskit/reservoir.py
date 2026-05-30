from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from .config import QRCConfig

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
except Exception:  # pragma: no cover
    QuantumCircuit = None  # type: ignore
    AerSimulator = None  # type: ignore
    NoiseModel = None  # type: ignore

class QRCReservoir:
    def __init__(self, cfg: QRCConfig):
        if QuantumCircuit is None:
            raise ImportError("qiskit is required (pip install qiskit qiskit-aer).")
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self._init_disorder()

    def _init_disorder(self) -> None:
        cfg = self.cfg
        n = cfg.total_qubits()
        rs = np.random.RandomState(cfg.seed)
        self.hx0 = 1.0 + 0.3 * rs.randn(n)
        self.hz1 = 1.0 + 0.3 * rs.randn(n)
        J = rs.rand(n, n)
        self.J = np.triu(J, 1)
        self.input_sign = rs.choice([-1.0, 1.0], size=n)

    def _apply_encoding(self, qc: "QuantumCircuit", uval: float) -> None:
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
        elif cfg.encoding == "hamiltonian_trotter":
            for q in range(n):
                sgn = 1.0 if cfg.input_map == "global" else float(self.input_sign[q])
                a = cfg.tau * (s * uval) * (self.hz1[q] * sgn)
                qc.rz(2.0 * a, q)
        else:
            raise ValueError(f"Unknown encoding: {cfg.encoding}")

    def _apply_reservoir_unitary(self, qc: "QuantumCircuit") -> None:
        cfg = self.cfg
        n = cfg.total_qubits()
        if cfg.reservoir_type == "ising_like":
            for _ in range(cfg.depth_per_step):
                for q in range(n):
                    a = cfg.tau * self.hx0[q] / cfg.depth_per_step
                    qc.rx(2.0 * a, q)
                for i in range(n):
                    for j in range(i + 1, n):
                        Jij = self.J[i, j]
                        if Jij != 0.0:
                            a = cfg.tau * Jij / cfg.depth_per_step
                            qc.rzz(2.0 * a, i, j)
        elif cfg.reservoir_type == "random_cx_rz":
            for _ in range(cfg.depth_per_step):
                perm = np.arange(n); self.rng.shuffle(perm)
                pairs = [(perm[k], perm[k+1]) for k in range(0, n-1, 2)]
                for a, b in pairs:
                    qc.cx(a, b); qc.rz(float(self.rng.uniform(-np.pi, np.pi)), b); qc.cx(a, b)
                for q in range(n):
                    qc.rz(float(self.rng.uniform(-np.pi, np.pi)), q)
        else:
            raise ValueError(f"Unknown reservoir_type: {cfg.reservoir_type}")

    def _apply_purification_entangle(self, qc: "QuantumCircuit") -> None:
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
        cfg = self.cfg
        nS, nA = cfg.n_system, cfg.n_ancilla
        n = cfg.total_qubits()
        T = len(inputs)
        sys_bits = nS if measure_system else 0
        anc_bits = nA if (cfg.use_purification and nA > 0) else 0
        qc = QuantumCircuit(n, T * (sys_bits + anc_bits))
        cidx = 0
        for uval in inputs:
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

    @staticmethod
    def _bit_at_from_right(bitstring: str, idx: int) -> int:
        return 1 if bitstring[-(idx + 1)] == "1" else 0

    @classmethod
    def _z_expectation_from_counts(cls, counts: Dict[str, int], shots: int, clbit_index: int) -> float:
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


# Backward-compatible alias retained inside the forked codebase.
NISQReservoir = QRCReservoir
