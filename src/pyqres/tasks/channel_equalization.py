from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Protocol, Sequence, Tuple

import numpy as np

from ..utils.linear import ridge_regression_fit, ridge_regression_predict


class ReservoirProtocol(Protocol):
    def run_stream(self, inputs: Sequence[float]) -> np.ndarray:
        ...


@dataclass
class ChannelEqualizationConfig:
    T_total: int = 3000
    washout: int = 200
    train_len: int = 1800
    test_len: int = 800
    delay: int = 2
    input_seed: int = 2026
    ridge_l2: float = 1e-6
    # Causal channel taps (current -> older samples)
    taps: Tuple[float, ...] = (0.08, 0.12, -0.18, -0.10)
    # Cubic nonlinearity coefficients
    nonlin2: float = 0.036
    nonlin3: float = -0.011
    noise_std: float = 0.02
    metric: Literal["ber", "mse"] = "ber"


@dataclass
class ChannelEqualizationDatasetConfig:
    n_train: int = 100
    n_test: int = 100
    n_symb: int = 100
    snr_db: float = 20.0
    input_seed: int = 17462
    symbols: Tuple[float, ...] = (-3.0, -1.0, 1.0, 3.0)
    taps: Tuple[float, ...] = (1.0, 0.18, -0.10, 0.091, -0.05, 0.04, 0.03, 0.01)
    nonlin2: float = 0.06
    nonlin3: float = -0.01


def _channel_response(message: np.ndarray, cfg: ChannelEqualizationDatasetConfig) -> tuple[np.ndarray, float]:
    message = np.asarray(message, dtype=float).reshape(-1)
    n_taps = len(cfg.taps)
    padded = np.concatenate((message[-n_taps:], message))
    linear = np.convolve(np.asarray(cfg.taps, dtype=float), padded, mode="full")[n_taps : -n_taps + 1]
    noise_scale = float(np.sqrt(10.0 ** (-float(cfg.snr_db) / 10.0)))
    return linear + cfg.nonlin2 * (linear**2) + cfg.nonlin3 * (linear**3), noise_scale


def generate_channel_equalization_dataset(cfg: ChannelEqualizationDatasetConfig) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.input_seed)

    def sample_split(n_messages: int) -> tuple[np.ndarray, np.ndarray]:
        messages = rng.choice(np.asarray(cfg.symbols, dtype=float), size=(n_messages, cfg.n_symb)).astype(float)
        observed = np.zeros_like(messages)
        for idx in range(n_messages):
            noiseless, noise_scale = _channel_response(messages[idx], cfg)
            observed[idx] = noiseless + noise_scale * rng.normal(size=cfg.n_symb)
        return messages, observed

    train_messages, train_observed = sample_split(cfg.n_train)
    test_messages, test_observed = sample_split(cfg.n_test)
    return {
        "train_messages": train_messages,
        "train_observed": train_observed,
        "test_messages": test_messages,
        "test_observed": test_observed,
        "symbols": np.asarray(cfg.symbols, dtype=float),
    }


def generate_channel_equalization_data(cfg: ChannelEqualizationConfig) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.input_seed)
    s = rng.choice([-1.0, 1.0], size=cfg.T_total).astype(float)
    maxlag = max(0, len(cfg.taps) - 1)

    v = np.zeros_like(s)
    for i, a in enumerate(cfg.taps):
        if i == 0:
            v += a * s
        else:
            v[i:] += a * s[:-i]

    y = v + cfg.nonlin2 * (v**2) + cfg.nonlin3 * (v**3)
    if cfg.noise_std > 0.0:
        y += rng.normal(0.0, cfg.noise_std, size=cfg.T_total)

    # Delay is clipped to guarantee a valid causal target index.
    d = int(np.clip(cfg.delay, 0, cfg.T_total - 1))
    target = np.zeros_like(s)
    target[d:] = s[:-d] if d > 0 else s
    # Early target positions are undefined due to delay; keep them but washout should discard.
    return y.astype(float), target.astype(float)


class ChannelEqualizationTaskRunner:
    def __init__(self, reservoir: ReservoirProtocol, cfg: ChannelEqualizationConfig):
        self.res = reservoir
        self.cfg = cfg

    def generate_io(self) -> Tuple[np.ndarray, np.ndarray]:
        return generate_channel_equalization_data(self.cfg)

    def run(self) -> Dict[str, float]:
        cfg = self.cfg
        u, target = self.generate_io()
        X = self.res.run_stream(u.tolist())

        t0 = max(cfg.washout, cfg.delay, len(cfg.taps) - 1)
        t_train_end = t0 + cfg.train_len
        t_test_end = t_train_end + cfg.test_len
        tr = np.arange(t0, t_train_end)
        te = np.arange(t_train_end, t_test_end)

        w = ridge_regression_fit(X[tr], target[tr], l2=cfg.ridge_l2)
        yhat_tr = ridge_regression_predict(X[tr], w)
        yhat_te = ridge_regression_predict(X[te], w)

        pred_tr = np.where(yhat_tr >= 0.0, 1.0, -1.0)
        pred_te = np.where(yhat_te >= 0.0, 1.0, -1.0)

        ber_tr = float(np.mean(pred_tr != target[tr]))
        ber_te = float(np.mean(pred_te != target[te]))
        mse_tr = float(np.mean((yhat_tr - target[tr]) ** 2))
        mse_te = float(np.mean((yhat_te - target[te]) ** 2))

        return {
            "train_ber": ber_tr,
            "test_ber": ber_te,
            "train_mse": mse_tr,
            "test_mse": mse_te,
            "train_score": -ber_tr if cfg.metric == "ber" else -mse_tr,
            "test_score": -ber_te if cfg.metric == "ber" else -mse_te,
        }


class ChannelEqualizationReservoirProtocol(Protocol):
    def run_stream(self, inputs: Sequence[float]) -> np.ndarray:
        ...

    def reset(self, rhoS0: np.ndarray | None = None) -> None:
        ...


def collect_channel_equalization_reservoir_features(
    reservoir: ChannelEqualizationReservoirProtocol,
    observed_messages: np.ndarray,
    initial_state: np.ndarray | None = None,
) -> np.ndarray:
    observed_messages = np.asarray(observed_messages, dtype=float)
    features = []
    for message in observed_messages:
        reservoir.reset(rhoS0=initial_state)
        message_features = np.asarray(reservoir.run_stream(message.tolist()), dtype=float)
        if message_features.shape[0] != message.shape[0]:
            raise ValueError("Reservoir feature length must match message length.")
        features.append(message_features)
    return np.stack(features, axis=0)
