"""Minimal global-Floquet partial-shadow memory run."""

from __future__ import annotations

import numpy as np

import pyqres as qres


def main() -> None:
    res = qres.qresreservoir.from_dict(
        {
            "preset": "prethermal_shadow",
            "memory_qubits": 4,
            "readout_qubits": 2,
            "backend": "exact",
            "floquet": {"omega": 12.0, "n_cycles_per_step": 4, "seed": 0},
            "encoding": {"mode": "prethermal_shadow", "input_qubits": "memory", "axis": "y", "scale": 0.12, "seed": 1},
            "shadow": {"pauli_k": 2, "shots": 2048, "measurement_type": "weak", "weak_strength": 0.5, "seed": 2},
            "reset": {"reset_state": "zero"},
            "simulator": {"seed_simulator": 3},
        }
    )
    inputs = np.random.default_rng(0).normal(size=20)
    features = res.run(inputs)
    lams = res.memory_channel_spectrum(pauli_k=1)
    print("feature_shape:", features.shape)
    print("feature_names:", res.get_feature_names())
    print("max_memory_lambda_abs:", float(np.max(np.abs(lams))))


if __name__ == "__main__":
    main()
