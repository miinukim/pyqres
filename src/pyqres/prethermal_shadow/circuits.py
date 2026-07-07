from __future__ import annotations

"""Circuit stubs for the global-Floquet partial-shadow reservoir.

The current implementation is intentionally dense density-matrix based. Qiskit
circuit construction for the all-qubit Floquet block can be added later without
changing the public dense reservoir API.
"""


def require_qiskit_circuit_backend() -> None:
    """Raise a clear error for circuit-backed execution."""

    raise NotImplementedError(
        "GlobalFloquetPartialShadowReservoir currently supports dense density-matrix "
        "simulation only; Qiskit circuit construction is not implemented yet."
    )


__all__ = ["require_qiskit_circuit_backend"]
