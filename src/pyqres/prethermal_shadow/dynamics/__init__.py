"""Focused building blocks for dense global-Floquet dynamics.

The public names formerly defined in one module remain available directly
from pyqres.prethermal_shadow.dynamics.
"""

from .hamiltonians import (
    build_drive_hamiltonian,
    build_fast_period_unitary,
    build_global_h0,
    build_step_unitary,
    drive_pauli_terms,
    fast_period,
    global_h0_pauli_terms,
    step_duration,
)
from .operators import (
    PAULI,
    PauliPlacements,
    PauliTerm,
    kron_all,
    parse_pauli_operator,
    parse_pauli_placements,
    parse_pauli_terms,
    pauli_string,
    pauli_terms_matrix,
    validate_axis,
)
from .parameters import (
    chain_edges,
    generate_hamiltonian_parameters,
    input_beta_array,
    resolve_input_qubits,
    resolve_qubit_partition,
    topology_edges,
    validate_floquet_config,
)
from .states import (
    combine_subsystem_states,
    density_plus,
    density_zero,
    partial_trace_memory_first,
    partial_trace_qubits,
    project_density,
    reorder_qubit_operator,
)

__all__ = [
    "PAULI",
    "PauliPlacements",
    "PauliTerm",
    "build_drive_hamiltonian",
    "build_fast_period_unitary",
    "build_global_h0",
    "build_step_unitary",
    "chain_edges",
    "combine_subsystem_states",
    "density_plus",
    "density_zero",
    "drive_pauli_terms",
    "fast_period",
    "generate_hamiltonian_parameters",
    "global_h0_pauli_terms",
    "input_beta_array",
    "kron_all",
    "parse_pauli_operator",
    "parse_pauli_placements",
    "parse_pauli_terms",
    "partial_trace_memory_first",
    "partial_trace_qubits",
    "pauli_string",
    "pauli_terms_matrix",
    "project_density",
    "reorder_qubit_operator",
    "resolve_input_qubits",
    "resolve_qubit_partition",
    "step_duration",
    "topology_edges",
    "validate_axis",
    "validate_floquet_config",
]

# Retain the original public module metadata for introspection and pickling.
for _public_name in __all__:
    _public_object = globals()[_public_name]
    if hasattr(_public_object, "__code__"):
        _public_object.__module__ = __name__

del _public_name, _public_object
