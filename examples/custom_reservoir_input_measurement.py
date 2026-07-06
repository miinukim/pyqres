"""Custom reservoir, input encoding, and measurement-control example.

This example intentionally avoids named presets. It builds a small dense
simulation reservoir from explicit Pauli Hamiltonian terms, prepares each input
by amplitude encoding on the ancilla qubit, and uses weak measurement with
measurement-conditioned feedback on the memory qubit.

Run from the repository root:

    python examples/custom_reservoir_input_measurement.py
"""

from __future__ import annotations

import numpy as np

import pyqres as qres
from pyqres.core.control import MeasurementControlConfig
from pyqres.core.reservoir_params import ReservoirParams
from pyqres.simulation import ChannelMapReservoir, ChannelMapReservoirConfig


def make_custom_reservoir() -> ChannelMapReservoir:
    """Build a non-preset reservoir with explicit dynamics and measurement."""

    # Joint register convention for Pauli terms:
    #   qubit 0: memory/system qubit
    #   qubit 1: readout/ancilla qubit
    hamiltonian = ReservoirParams.from_pauli_terms(
        n_system=1,
        n_ancilla=1,
        h0_terms=[
            (0.45, ((0, "X"),)),
            (0.20, ((1, "X"),)),
            (0.70, ((0, "Z"), (1, "Z"))),
        ],
        h1_terms=[
            (0.90, ((0, "Z"),)),
            (-0.35, ((1, "Z"),)),
        ],
        tau=0.8,
        seed=7,
    ).generate()

    control = MeasurementControlConfig(
        measurement_mode="weak",
        measurement_strength=0.65,
        post_measurement_mode="reset",
        conditioned_gate="system_rz",
        conditioned_gate_angle=0.25 * np.pi,
        conditioned_gate_target=0,
        conditioned_gate_condition="nonzero",
    )

    cfg = ChannelMapReservoirConfig(
        n_system=1,
        n_ancilla=1,
        tau=hamiltonian["tau"],
        H0_hamiltonian=hamiltonian["H0_hamiltonian"],
        H1_hamiltonian=hamiltonian["H1_hamiltonian"],
        # Use amplitude encoding instead of Hamiltonian modulation. Inputs are
        # mapped to [0, 1] before preparing the selected ancilla qubit.
        input_encoding="amplitude",
        input_scale=0.5,
        input_bias=0.5,
        encoding_register="ancilla",
        encoding_targets=(0,),
        amplitude_encoding_style="sqrt_u_sqrt_1_minus_u",
        input_unitary_order="before",
        include_bias=True,
        use_shot_noise=False,
        init_state="zero",
        control=control,
        seed=7,
    )
    return ChannelMapReservoir(cfg)


def main() -> None:
    rng = np.random.default_rng(12)
    steps = np.linspace(0.0, 10.0, 180)

    # Any task can supply its own stream. Here the input is normalized to
    # roughly [-1, 1], then the reservoir config maps it into [0, 1].
    inputs = np.sin(steps) + 0.15 * rng.standard_normal(steps.shape)
    inputs = np.clip(inputs, -1.0, 1.0)
    targets = np.roll(inputs, -1)

    dataset = qres.data.arrays(inputs[:-1], targets[:-1]).split(
        washout=20,
        train=110,
        test=48,
    )

    result = qres.Experiment(
        reservoir=make_custom_reservoir(),
        dataset=dataset,
        readout=qres.Ridge(l2=1e-5),
        metrics=["mse", "r2"],
        metadata={"example": "custom_reservoir_input_measurement"},
    ).run()

    print("feature_shape:", result.features.shape)
    print("train:", result.metrics["train"])
    print("test:", result.metrics["test"])


if __name__ == "__main__":
    main()
