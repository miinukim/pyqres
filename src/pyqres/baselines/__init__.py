"""Public classical baseline and readout API.

This module collects the baseline implementations into a compact namespace so
experiment code can import `pyqres.baselines` without knowing whether a model is
implemented in `classical.py` or `esn.py`.
"""

from .classical import (
    LogisticEqualizerConfig,
    SoftmaxReadoutConfig,
    fit_softmax_readout,
    predict_softmax_readout,
    run_channel_equalization_logistic,
    run_channel_equalization_symbol_logistic,
)
from .esn import ESNConfig, EchoStateNetwork, run_stm_esn, run_channel_equalization_esn

__all__ = [
    "ESNConfig",
    "EchoStateNetwork",
    "LogisticEqualizerConfig",
    "SoftmaxReadoutConfig",
    "fit_softmax_readout",
    "predict_softmax_readout",
    "run_stm_esn",
    "run_channel_equalization_esn",
    "run_channel_equalization_logistic",
    "run_channel_equalization_symbol_logistic",
]
