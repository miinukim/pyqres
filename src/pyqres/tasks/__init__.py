"""Public benchmark-task API.

Task runners are intentionally backend-neutral: they expect reservoir objects
with the small streaming protocol, so exact, Qiskit, or future reservoirs can be
swapped into the same benchmark code.
"""

from .stm import STMConfig, STMTaskRunner
from .mackey_glass import MackeyGlassConfig, MackeyGlassTaskRunner, generate_mackey_glass_series
from .channel_equalization import (
    ChannelEqualizationConfig,
    ChannelEqualizationDatasetConfig,
    ChannelEqualizationReservoirProtocol,
    ChannelEqualizationTaskRunner,
    collect_channel_equalization_reservoir_features,
    generate_channel_equalization_data,
    generate_channel_equalization_dataset,
)

__all__ = [
    "STMConfig",
    "STMTaskRunner",
    "MackeyGlassConfig",
    "MackeyGlassTaskRunner",
    "generate_mackey_glass_series",
    "ChannelEqualizationConfig",
    "ChannelEqualizationDatasetConfig",
    "ChannelEqualizationReservoirProtocol",
    "ChannelEqualizationTaskRunner",
    "collect_channel_equalization_reservoir_features",
    "generate_channel_equalization_data",
    "generate_channel_equalization_dataset",
]
