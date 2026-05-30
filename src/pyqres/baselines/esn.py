from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Literal, Sequence
import numpy as np
from ..utils.linear import ridge_regression_fit, ridge_regression_predict, rmse, r2_score

@dataclass
class ESNConfig:
    n_res: int = 200
    spectral_radius: float = 0.9
    input_scale: float = 0.5
    leak_rate: float = 1.0
    ridge_l2: float = 1e-6
    seed: int = 7
    state_clip: float = 5.0
    power_iter: int = 200

'''
def _spectral_radius_power_iter(A: np.ndarray, seed: int, n_iter: int = 200) -> float:
    """
    Overflow-safe spectral radius estimate using power iteration on a scaled matrix.
    Returns 0.0 if estimation fails.
    """
    rng = np.random.default_rng(seed)

    # Basic sanity
    if not np.isfinite(A).all():
        return 0.0

    # Scale A down to avoid overflow in intermediate matmul
    amax = float(np.max(np.abs(A)))
    if not np.isfinite(amax) or amax <= 0.0:
        return 0.0
    As = A / amax

    v = rng.normal(size=(As.shape[0],)).astype(np.float64)
    v /= (np.linalg.norm(v) + 1e-12)

    for _ in range(n_iter):
        w = As @ v
        if not np.isfinite(w).all():
            return 0.0
        nrm = float(np.linalg.norm(w))
        if not np.isfinite(nrm) or nrm < 1e-12:
            return 0.0
        v = w / nrm

    if not np.isfinite(As).all():
        raise FloatingPointError(f"As has non-finite values. amax={amax}, A finite={np.isfinite(A).all()}")

    if not np.isfinite(v).all():
        raise FloatingPointError("v has non-finite values before matmul.")

    # dtype checks
    if As.dtype != np.float64:
        As = As.astype(np.float64, copy=False)
    if v.dtype != np.float64:
        v = v.astype(np.float64, copy=False)
    w = As @ v
    if not np.isfinite(w).all():
        return 0.0
    sr_scaled = float(np.linalg.norm(w))
    if not np.isfinite(sr_scaled):
        return 0.0

    # Undo scaling
    return sr_scaled * amax
'''


class EchoStateNetwork:
    def __init__(self, cfg: ESNConfig):
        self.cfg = cfg
        if not (0.0 <= cfg.leak_rate <= 1.0):
            raise ValueError("ESNConfig.leak_rate must be in [0,1].")
        if cfg.spectral_radius <= 0:
            raise ValueError("ESNConfig.spectral_radius must be > 0.")
        rng = np.random.default_rng(cfg.seed)

        n = cfg.n_res
        # Generate random reservoir
        W = rng.normal(0.0, 1.0, size=(n, n)).astype(np.float64)

        # Gershgorin / infinity-norm bound (safe upper bound)
        row_sum = np.sum(np.abs(W), axis=1)
        rho_bound = float(np.max(row_sum))

        if not np.isfinite(rho_bound) or rho_bound <= 1e-12:
            rho_bound = 1.0

        # Scale to desired spectral radius
        W *= (cfg.spectral_radius / rho_bound)

        self.W_res = W

        self.W_in = cfg.input_scale * rng.normal(0.0, 1.0, size=(n, 2)).astype(np.float64)
        self.x = np.zeros(n, dtype=np.float64)

    def step(self, u: float) -> np.ndarray:
        cfg = self.cfg
        inp = np.array([1.0, u], dtype=np.float64)
        # Some BLAS backends can raise spurious divide/underflow flags in matmul when
        # global np.seterr(all="raise") is enabled. Ignore flags locally and validate
        # finiteness explicitly below.
        with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
            pre = self.W_res @ self.x + self.W_in @ inp

        if not np.isfinite(pre).all():
            raise FloatingPointError("ESN pre-activation became non-finite. Reduce spectral_radius/input_scale.")

        x_new = np.tanh(pre)
        self.x = (1.0 - cfg.leak_rate) * self.x + cfg.leak_rate * x_new

        # clip as safety valve
        if cfg.state_clip is not None and cfg.state_clip > 0:
            self.x = np.clip(self.x, -cfg.state_clip, cfg.state_clip)

        if not np.isfinite(self.x).all():
            raise FloatingPointError("ESN state became non-finite. Reduce spectral_radius/input_scale.")
        return self.x.copy()

    def collect_states(self, u_seq: Sequence[float]) -> np.ndarray:
        X = []
        for u in u_seq:
            X.append(self.step(float(u)))
        X = np.vstack(X)
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        if not np.isfinite(X).all():
            raise FloatingPointError("ESN produced non-finite states matrix X.")
        return X

def run_stm_esn(T_total: int, washout: int, train_len: int, test_len: int, delays: Sequence[int],
                input_dist: Literal["uniform_pm1","gaussian"], input_seed: int,
                esn_cfg: ESNConfig, metric: Literal["r2","rmse"]="r2") -> Dict[int, Dict[str, float]]:
    rng = np.random.default_rng(input_seed)
    if input_dist == "uniform_pm1":
        u = rng.choice([-1.0, 1.0], size=T_total).astype(float)
    else:
        u = rng.normal(0.0, 1.0, size=T_total).astype(float)
    esn = EchoStateNetwork(esn_cfg)
    X = esn.collect_states(u.tolist())
    t0 = washout
    t_train_end = t0 + train_len
    t_test_end = t_train_end + test_len
    out: Dict[int, Dict[str, float]] = {}
    for d in delays:
        t_start = max(t0, d)
        tr = np.arange(t_start, t_train_end)
        te = np.arange(t_train_end, t_test_end)
        y_tr = u[tr - d]
        y_te = u[te - d]
        w = ridge_regression_fit(X[tr], y_tr, l2=esn_cfg.ridge_l2)
        yhat_tr = ridge_regression_predict(X[tr], w)
        yhat_te = ridge_regression_predict(X[te], w)
        if metric == "r2":
            out[d] = {"train_score": float(r2_score(y_tr, yhat_tr)),
                      "test_score": float(r2_score(y_te, yhat_te))}
        else:
            out[d] = {"train_score": float(-rmse(yhat_tr, y_tr)),
                      "test_score": float(-rmse(yhat_te, y_te))}
    return out


def run_channel_equalization_esn(
    observed: np.ndarray,
    target: np.ndarray,
    washout: int,
    train_len: int,
    test_len: int,
    esn_cfg: ESNConfig,
    ridge_l2: float = 1e-6,
    metric: Literal["ber", "mse"] = "ber",
) -> Dict[str, float]:
    observed = np.asarray(observed, dtype=float).reshape(-1)
    target = np.asarray(target, dtype=float).reshape(-1)
    if observed.shape[0] != target.shape[0]:
        raise ValueError("observed and target must have the same length.")
    if metric not in {"ber", "mse"}:
        raise ValueError("metric must be 'ber' or 'mse'.")

    esn = EchoStateNetwork(esn_cfg)
    X = esn.collect_states(observed.tolist())
    t0 = int(washout)
    t_train_end = t0 + int(train_len)
    t_test_end = t_train_end + int(test_len)
    tr = np.arange(t0, t_train_end)
    te = np.arange(t_train_end, t_test_end)

    w = ridge_regression_fit(X[tr], target[tr], l2=ridge_l2)
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
        "train_score": -ber_tr if metric == "ber" else -mse_tr,
        "test_score": -ber_te if metric == "ber" else -mse_te,
    }
