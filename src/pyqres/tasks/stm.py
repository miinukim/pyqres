"""Short-term memory benchmark task."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Literal, Protocol, Sequence
import numpy as np

from ..utils.linear import ridge_regression_fit, ridge_regression_predict, rmse, r2_score


class ReservoirProtocol(Protocol):
    """Reservoir interface required by the STM task runner."""

    def run_stream(self, inputs: Sequence[float]) -> np.ndarray:
        ...

@dataclass
class STMConfig:
    """Configuration for the STM benchmark stream and readout split."""

    T_total: int = 4000
    washout: int = 500
    train_len: int = 2000
    test_len: int = 1000
    delays: Sequence[int] = tuple(range(1, 41))
    input_dist: Literal["uniform_pm1", "gaussian"] = "uniform_pm1"
    input_seed: int = 2026
    ridge_l2: float = 1e-6
    metric: Literal["r2", "rmse"] = "r2"

class STMTaskRunner:
    """Generate STM data, collect reservoir features, and score delayed recall."""

    def __init__(self, reservoir: ReservoirProtocol, cfg: STMConfig):
        self.res = reservoir
        self.cfg = cfg

    def generate_inputs(self) -> np.ndarray:
        """Generate the scalar input stream used as delayed targets."""

        rng = np.random.default_rng(self.cfg.input_seed)
        if self.cfg.input_dist == "uniform_pm1":
            u = rng.choice([-1.0, 1.0], size=self.cfg.T_total)
        else:
            u = rng.normal(0.0, 1.0, size=self.cfg.T_total)
        return u.astype(float)

    def run(self) -> Dict[int, Dict[str, float]]:
        """Fit one linear readout per delay and return train/test scores."""

        cfg = self.cfg
        u = self.generate_inputs()
        X = self.res.run_stream(u.tolist())
        t0 = cfg.washout
        t_train_end = t0 + cfg.train_len
        t_test_end = t_train_end + cfg.test_len

        out: Dict[int, Dict[str, float]] = {}
        for d in cfg.delays:
            # The first valid sample for delay d is t=d. Washout may impose a
            # later start, so choose the stricter boundary.
            t_start = max(t0, d)
            tr = np.arange(t_start, t_train_end)
            te = np.arange(t_train_end, t_test_end)
            y_tr = u[tr - d]
            y_te = u[te - d]
            w = ridge_regression_fit(X[tr], y_tr, l2=cfg.ridge_l2)
            yhat_tr = ridge_regression_predict(X[tr], w)
            yhat_te = ridge_regression_predict(X[te], w)
            if cfg.metric == "r2":
                out[d] = {"train_score": float(r2_score(y_tr, yhat_tr)),
                          "test_score": float(r2_score(y_te, yhat_te))}
            else:
                out[d] = {"train_score": float(-rmse(yhat_tr, y_tr)),
                          "test_score": float(-rmse(yhat_te, y_te))}
        return out

    @staticmethod
    def memory_capacity(results: Dict[int, Dict[str, float]], use_test: bool=True) -> float:
        """Sum positive delay scores as a standard memory-capacity proxy."""

        key = "test_score" if use_test else "train_score"
        return float(sum(max(0.0, v[key]) for v in results.values()))
