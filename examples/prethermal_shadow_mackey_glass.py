"""Run a small prethermal shadow reservoir forecasting example.

This example prefers pyqres-tasks' Mackey-Glass dataset when available. If that
package is not installed, it falls back to a short sinusoid so the example can
still be run from this repository.

Advanced users can pass custom ``basis_sampler``, ``schedule_executor``, or
``feature_builder`` callables to ``PrethermalShadowReservoir`` to replace the
default random schedules, Aer execution, or shadow reconstruction.
"""

from __future__ import annotations

import numpy as np

import pyqres as qres
from pyqres.experimental.prethermal_shadow import (
    InputWriteConfig,
    PrethermalFloquetConfig,
    PrethermalShadowConfig,
    PrethermalShadowReservoir,
    ShadowReadoutConfig,
    TransducerConfig,
)


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
    cfg = PrethermalShadowConfig(
        n_readout=2,
        floquet=PrethermalFloquetConfig(n_memory=3, n_floquet=2, tau=0.15, seed=23),
        input_write=InputWriteConfig(axis="y", beta=0.08, bias=0.0),
        transducer=TransducerConfig(tau_c=0.04, seed=29),
        shadow=ShadowReadoutConfig(pauli_k=2, shots=12, include_bias=True, seed=31),
        simulator_method="density_matrix",
        seed_simulator=37,
    )
    reservoir = PrethermalShadowReservoir(cfg)
    dataset = make_dataset()
    result = qres.Experiment(
        reservoir=reservoir,
        dataset=dataset,
        readout=qres.Ridge(l2=1e-6),
        metrics=["mse", "r2"],
        metadata={"example": "prethermal_shadow_mackey_glass"},
    ).run()
    print("feature_shape:", result.features.shape)
    print("feature_labels:", reservoir.feature_labels)
    print("metrics:", result.metrics)


if __name__ == "__main__":
    main()
