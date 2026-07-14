"""Small Mackey-Glass example for global-Floquet partial-shadow QRC."""

from __future__ import annotations

import numpy as np

import pyqres as qres


def make_dataset():
    try:
        from pyqres_tasks import MackeyGlassConfig, mackey_glass_dataset

        return mackey_glass_dataset(
            MackeyGlassConfig(T_total=34, washout=6, train_len=18, test_len=8, prediction_horizon=1)
        )
    except Exception:
        series = np.sin(np.linspace(0.0, 8.0, 35))
        return qres.data.timeseries(series, target_horizon=1).split(washout=6, train=18, test=8)


def main() -> None:
    reservoir = qres.qresreservoir.from_dict(
        {
            "preset": "prethermal_shadow",
            "memory_qubits": 3,
            "readout_qubits": 2,
            "backend": "exact",
            "floquet": {"omega": 16.0, "n_cycles_per_step": 2, "seed": 23},
            "encoding": {"mode": "prethermal_shadow", "input_qubits": "memory", "axis": "y", "scale": 0.08, "seed": 29},
            "shadow": {"pauli_k": 2, "shots": 64, "include_bias": True, "measurement_type": "projective", "seed": 31},
            "reset": {"reset_state": "zero"},
            "simulator": {"seed_simulator": 37},
        }
    )
    dataset = make_dataset()
    result = qres.Experiment(
        reservoir=reservoir,
        dataset=dataset,
        readout=qres.Ridge(l2=1e-6),
        metrics=["mse", "r2"],
        metadata={"example": "global_floquet_partial_shadow_mackey_glass"},
    ).run()
    print("feature_shape:", result.features.shape)
    print("feature_labels:", reservoir.get_feature_names())
    print("metrics:", result.metrics)


if __name__ == "__main__":
    main()
