from __future__ import annotations

"""Small public preset registry for common reservoir specs."""

from .specs import ReadoutSpec, ReservoirSpec


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
        n_memory=n_memory,
        n_readout=n_readout,
        tau=tau,
        seed=seed,
        readout=ReadoutSpec(mode="memory_observables", observables=observables, count=observable_count, init_state="zero"),
    )


def names() -> list[str]:
    """List available built-in preset names."""

    return ["ising.memory_readout"]


def get(name: str, **kwargs: object) -> ReservoirSpec:
    """Instantiate a named preset."""

    key = name.lower()
    if key == "ising.memory_readout":
        return ising_memory_readout(**kwargs)
    raise ValueError(f"Unknown preset '{name}'. Available presets: {names()}")
