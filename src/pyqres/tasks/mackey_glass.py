"""Mackey-Glass time-series forecasting benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Protocol, Sequence, Tuple

import numpy as np

from ..utils.linear import ridge_regression_fit, ridge_regression_predict, rmse, r2_score


class ReservoirProtocol(Protocol):
    """Reservoir interface required by the Mackey-Glass task runner."""

    def run_stream(self, inputs: Sequence[float]) -> np.ndarray:
        ...


@dataclass
class MackeyGlassConfig:
    """Configuration for Mackey-Glass generation and readout scoring."""

    T_total: int = 1200
    washout: int = 200
    train_len: int = 600
    test_len: int = 300
    prediction_horizon: int = 1
    input_seed: int = 2026
    ridge_l2: float = 1e-6
    metric: Literal["r2", "rmse"] = "r2"
    beta: float = 0.2
    gamma: float = 0.1
    delay: float = 17.0
    power: float = 10.0
    dt: float = 1.0
    warmup_steps: int = 200
    history_init: float = 1.2


def generate_mackey_glass_series(cfg: MackeyGlassConfig) -> np.ndarray:
    """Generate a scalar Mackey-Glass sequence by Euler integration."""

    horizon = int(cfg.prediction_horizon)
    if horizon < 1:
        raise ValueError(f"prediction_horizon must be >= 1, got {horizon}.")
    delay_steps = int(round(float(cfg.delay) / float(cfg.dt)))
    if delay_steps < 1:
        raise ValueError(f"delay/dt must correspond to at least one step, got {delay_steps}.")

    total_steps = int(cfg.T_total) + horizon + int(cfg.warmup_steps)
    history = np.full(delay_steps + 1, float(cfg.history_init), dtype=float)
    series = np.zeros(total_steps + delay_steps + 1, dtype=float)
    series[: delay_steps + 1] = history

    for t in range(delay_steps, total_steps + delay_steps):
        # Discrete Euler update for dx/dt = beta*x_tau/(1+x_tau^p)-gamma*x.
        x_t = series[t]
        x_tau = series[t - delay_steps]
        dx = cfg.beta * x_tau / (1.0 + x_tau ** cfg.power) - cfg.gamma * x_t
        series[t + 1] = x_t + cfg.dt * dx

    trimmed = series[delay_steps + int(cfg.warmup_steps) : delay_steps + int(cfg.warmup_steps) + int(cfg.T_total) + horizon]
    return trimmed.astype(float)


class MackeyGlassTaskRunner:
    """Forecast future Mackey-Glass values from reservoir features."""

    def __init__(self, reservoir: ReservoirProtocol, cfg: MackeyGlassConfig):
        self.res = reservoir
        self.cfg = cfg

    def generate_io(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return aligned input and prediction-horizon target sequences."""

        series = generate_mackey_glass_series(self.cfg)
        horizon = int(self.cfg.prediction_horizon)
        inputs = series[:-horizon]
        targets = series[horizon:]
        return inputs.astype(float), targets.astype(float)

    def run(self) -> Dict[str, float]:
        """Fit a ridge readout and report train/test forecasting scores."""

        cfg = self.cfg
        u, target = self.generate_io()
        X = self.res.run_stream(u.tolist())

        t0 = cfg.washout
        t_train_end = t0 + cfg.train_len
        t_test_end = t_train_end + cfg.test_len
        tr = np.arange(t0, t_train_end)
        te = np.arange(t_train_end, t_test_end)

        w = ridge_regression_fit(X[tr], target[tr], l2=cfg.ridge_l2)
        yhat_tr = ridge_regression_predict(X[tr], w)
        yhat_te = ridge_regression_predict(X[te], w)

        if cfg.metric == "r2":
            train_score = float(r2_score(target[tr], yhat_tr))
            test_score = float(r2_score(target[te], yhat_te))
        else:
            train_score = float(-rmse(yhat_tr, target[tr]))
            test_score = float(-rmse(yhat_te, target[te]))

        return {
            "train_score": train_score,
            "test_score": test_score,
        }
