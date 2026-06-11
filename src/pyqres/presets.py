from __future__ import annotations

"""Small public preset registry for common reservoir specs."""

from pyqres.core.specs import ReadoutSpec, ReservoirSpec


def ising_memory_readout(
    *,
    n_memory: int = 4,
    n_readout: int = 1,
    tau: float = 1.0,
    observables: str = "z",
    observable_count: int | None = None,
    seed: int = 17462,
) -> ReservoirSpec:
    """Return a basic Ising memory/readout reservoir spec."""

    return ReservoirSpec(
        family="ising",
        preset="ising",
        source_kind="preset",
        n_memory=n_memory,
        n_readout=n_readout,
        tau=tau,
        seed=seed,
        readout=ReadoutSpec(mode="memory_observables", observables=observables, count=observable_count, init_state="zero"),
    )


def random_pauli_memory_readout(
    *,
    n_memory: int = 5,
    n_readout: int = 1,
    depth: int = 3,
    seed: int = 1234,
    observables: str = "z",
    observable_count: int | None = None,
) -> ReservoirSpec:
    """Return a RandomPauli dimension-model reservoir spec."""

    return ReservoirSpec(
        family="random_pauli",
        preset="random_pauli",
        source_kind="preset",
        n_memory=n_memory,
        n_readout=n_readout,
        seed=seed,
        model_kwargs={"depth": depth},
        readout=ReadoutSpec(mode="memory_observables", observables=observables, count=observable_count, init_state="zero"),
    )


def syk_memory_readout(
    *,
    n_memory: int = 7,
    n_readout: int = 1,
    tau: float = 1.0,
    seed: int = 1234,
    observables: str = "occupation",
    observable_count: int | None = None,
) -> ReservoirSpec:
    """Return an SYK dimension-model reservoir spec."""

    return ReservoirSpec(
        family="syk",
        preset="syk",
        source_kind="preset",
        n_memory=n_memory,
        n_readout=n_readout,
        tau=tau,
        seed=seed,
        readout=ReadoutSpec(mode="memory_observables", observables=observables, count=observable_count, init_state="zero"),
    )


def names() -> list[str]:
    """List available built-in preset names."""

    return ["ising.memory_readout", "random_pauli.memory_readout", "syk.memory_readout"]


def get(name: str, **kwargs: object) -> ReservoirSpec:
    """Instantiate a named preset."""

    key = name.lower()
    if key == "ising.memory_readout":
        return ising_memory_readout(**kwargs)
    if key == "random_pauli.memory_readout":
        return random_pauli_memory_readout(**kwargs)
    if key == "syk.memory_readout":
        return syk_memory_readout(**kwargs)
    raise ValueError(f"Unknown preset '{name}'. Available presets: {names()}")
