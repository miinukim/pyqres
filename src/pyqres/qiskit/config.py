"""Configuration dataclasses for Qiskit-backed reservoirs."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal, Optional
import numpy as np

from pyqres.core.control import MeasurementControlConfig

try:
    from qiskit_aer.noise import NoiseModel, amplitude_damping_error, phase_damping_error, depolarizing_error
except Exception:  # pragma: no cover
    NoiseModel = None  # type: ignore
    amplitude_damping_error = None  # type: ignore
    phase_damping_error = None  # type: ignore
    depolarizing_error = None  # type: ignore

EncodingType = Literal["rz_global", "rz_per_qubit", "hamiltonian_trotter"]
ReservoirType = Literal["ising_like", "random_cx_rz", "pauli_evolution"]
EvolutionSynthesisType = Literal["default", "lie_trotter", "suzuki_trotter"]
ReadoutType = Literal["z_local", "z_local_plus_anc", "pauli_k_local"]

@dataclass
class NoiseConfig:
    """Simple Aer noise-model builder.

    The parameters are intentionally backend-agnostic experiment knobs. They are
    converted to Qiskit Aer errors only when to_noise_model() is called, so the
    rest of the package can import configs even without Qiskit installed.
    """

    use_damping: bool = True
    dt: float = 1.0
    T1: Optional[float] = 100.0
    T2: Optional[float] = 100.0
    use_depolarizing: bool = False
    p_depol_1q: float = 0.0
    p_depol_2q: float = 0.0

    def to_noise_model(self) -> "NoiseModel":
        """Construct a Qiskit Aer NoiseModel from damping/depolarizing settings."""

        if NoiseModel is None:
            raise ImportError("qiskit-aer is required for noise models.")
        nm = NoiseModel()

        if self.use_damping:
            # Convert T1/T2-style times into amplitude- and phase-damping
            # probabilities for one logical circuit time step.
            p_amp = 0.0 if not self.T1 or self.T1 <= 0 else 1.0 - float(np.exp(-self.dt / self.T1))
            if not self.T2 or self.T2 <= 0:
                p_ph = 0.0
            else:
                inv_Tphi = max(0.0, 1.0 / self.T2 - 1.0 / (2.0 * max(self.T1 or 1e9, 1e-9)))
                Tphi = (1.0 / inv_Tphi) if inv_Tphi > 0 else 1e18
                p_ph = 1.0 - float(np.exp(-self.dt / Tphi))

            amp_err = amplitude_damping_error(p_amp)
            ph_err = phase_damping_error(p_ph)
            combined = amp_err.compose(ph_err)

            for g in ["rx", "rz", "x", "sx", "id", "u", "u3", "u2", "u1"]:
                nm.add_all_qubit_quantum_error(combined, g)

        if self.use_depolarizing:
            if self.p_depol_1q > 0:
                dep1 = depolarizing_error(self.p_depol_1q, 1)
                for g in ["rx", "rz", "x", "sx", "id", "u", "u3", "u2", "u1"]:
                    nm.add_all_qubit_quantum_error(dep1, g)
            if self.p_depol_2q > 0:
                dep2 = depolarizing_error(self.p_depol_2q, 2)
                for g in ["cx", "cz", "rzz", "swap"]:
                    nm.add_all_qubit_quantum_error(dep2, g)
        return nm

@dataclass
class QRCConfig:
    """Circuit-reservoir configuration used by QRCReservoir."""

    n_system: int = 4
    n_ancilla: int = 2
    reservoir_type: ReservoirType = "ising_like"
    H0_hamiltonian: Optional[Any] = None
    H1_hamiltonian: Optional[Any] = None
    evolution_synthesis: EvolutionSynthesisType = "lie_trotter"
    evolution_reps: int = 1
    evolution_order: int = 2
    evolution_insert_barriers: bool = False
    evolution_preserve_order: bool = True
    depth_per_step: int = 1
    tau: float = 1.0
    seed: int = 1234
    encoding: EncodingType = "hamiltonian_trotter"
    input_scale: float = 1.0
    input_map: Literal["global", "per_qubit_random_sign"] = "global"
    use_purification: bool = True
    ancilla_pattern: Literal["star", "pairwise"] = "star"
    measure_and_reset_ancilla: bool = True
    readout: ReadoutType = "z_local_plus_anc"
    pauli_k: int = 2
    include_bias: bool = True
    shots: int = 2048
    simulator_method: Literal["automatic", "density_matrix", "statevector"] = "density_matrix"
    transpile_optimization_level: int = 1
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    control: MeasurementControlConfig = field(default_factory=MeasurementControlConfig)

    def total_qubits(self) -> int:
        """Return system plus ancilla qubit count."""

        return self.n_system + self.n_ancilla


# Backward-compatible alias retained inside the forked codebase.
NISQRCConfig = QRCConfig
