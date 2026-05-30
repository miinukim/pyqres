from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal

import numpy as np
from scipy.optimize import minimize


@dataclass
class LogisticEqualizerConfig:
    n_lags: int = 8
    l2: float = 1e-4
    max_iter: int = 50
    tol: float = 1e-8


@dataclass
class SoftmaxReadoutConfig:
    fit_intercept: bool = True
    l2: float = 1e-6
    max_iter: int = 1000
    tol: float = 1e-9


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def _lagged_design_matrix(observed: np.ndarray, n_lags: int) -> np.ndarray:
    observed = np.asarray(observed, dtype=float).reshape(-1)
    if n_lags < 1:
        raise ValueError("n_lags must be at least 1.")
    T = observed.shape[0]
    X = np.ones((T, n_lags + 1), dtype=float)
    X[:, 1] = observed
    for lag in range(1, n_lags):
        X[lag:, lag + 1] = observed[:-lag]
    return X


def _message_lagged_design(observed_messages: np.ndarray, n_lags: int) -> np.ndarray:
    observed_messages = np.asarray(observed_messages, dtype=float)
    return np.stack([_lagged_design_matrix(message, n_lags) for message in observed_messages], axis=0)


def _prepare_features(X: np.ndarray, fit_intercept: bool) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if fit_intercept:
        return np.hstack([np.ones((X.shape[0], 1), dtype=float), X])
    return X


def fit_softmax_readout(X: np.ndarray, y: np.ndarray, cfg: SoftmaxReadoutConfig) -> Dict[str, Any]:
    X = _prepare_features(X, cfg.fit_intercept)
    y = np.asarray(y).reshape(-1)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if not np.isfinite(X).all():
        raise FloatingPointError("Non-finite values in softmax readout features.")

    classes = np.unique(y)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    y_idx = np.array([class_to_idx[val] for val in y], dtype=int)
    Y = np.eye(classes.size, dtype=float)[y_idx]

    n_samples, n_features = X.shape
    n_classes = classes.size

    reg_mask = np.ones((n_features, n_classes), dtype=float)
    if cfg.fit_intercept:
        reg_mask[0, :] = 0.0

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        W = theta.reshape(n_features, n_classes)
        with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
            scores = X @ W
        if not np.isfinite(scores).all():
            raise FloatingPointError("Softmax readout scores became non-finite.")
        scores -= np.max(scores, axis=1, keepdims=True)
        with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        if not np.isfinite(probs).all():
            raise FloatingPointError("Softmax readout probabilities became non-finite.")
        loss = -np.sum(Y * np.log(np.clip(probs, 1e-12, 1.0))) / n_samples
        loss += 0.5 * cfg.l2 * np.sum((W * reg_mask) ** 2)
        with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
            grad = (X.T @ (probs - Y)) / n_samples + cfg.l2 * W * reg_mask
        if not np.isfinite(grad).all():
            raise FloatingPointError("Softmax readout gradient became non-finite.")
        return float(loss), grad.reshape(-1)

    init = np.zeros(n_features * n_classes, dtype=float)
    result = minimize(
        fun=lambda theta: objective(theta)[0],
        x0=init,
        jac=lambda theta: objective(theta)[1],
        method="L-BFGS-B",
        options={"maxiter": int(cfg.max_iter), "ftol": float(cfg.tol)},
    )
    if (not result.success) and ("TOTAL NO. OF ITERATIONS REACHED LIMIT" not in str(result.message)):
        raise RuntimeError(f"Softmax readout optimization failed: {result.message}")
    if not np.isfinite(result.x).all():
        raise FloatingPointError("Softmax readout optimizer returned non-finite weights.")

    return {
        "weights": result.x.reshape(n_features, n_classes),
        "classes": classes,
        "fit_intercept": cfg.fit_intercept,
    }


def predict_softmax_readout(X: np.ndarray, model: Dict[str, Any]) -> np.ndarray:
    X = _prepare_features(X, bool(model["fit_intercept"]))
    weights = np.asarray(model["weights"], dtype=float)
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        scores = X @ weights
    if not np.isfinite(scores).all():
        raise FloatingPointError("Softmax readout prediction scores became non-finite.")
    pred_idx = np.argmax(scores, axis=1)
    return np.asarray(model["classes"])[pred_idx]


def _fit_logistic_regression(X: np.ndarray, y01: np.ndarray, cfg: LogisticEqualizerConfig) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y01 = np.asarray(y01, dtype=float).reshape(-1)
    if not np.isfinite(X).all() or not np.isfinite(y01).all():
        raise FloatingPointError("Non-finite values in logistic regression inputs.")
    w = np.zeros(X.shape[1], dtype=float)
    reg = cfg.l2 * np.eye(X.shape[1], dtype=float)
    reg[0, 0] = 0.0

    for _ in range(cfg.max_iter):
        with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
            logits = X @ w
        if not np.isfinite(logits).all():
            raise FloatingPointError("Logistic regression logits became non-finite.")
        probs = np.clip(_sigmoid(logits), 1e-9, 1.0 - 1e-9)
        with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
            grad = X.T @ (probs - y01) + reg @ w
        r = probs * (1.0 - probs)
        with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
            hessian = (X.T * r) @ X + reg
        if not np.isfinite(grad).all() or not np.isfinite(hessian).all():
            raise FloatingPointError("Logistic regression normal equations became non-finite.")
        step = np.linalg.solve(hessian, grad)
        if not np.isfinite(step).all():
            raise FloatingPointError("Logistic regression Newton step became non-finite.")
        w_new = w - step
        if np.linalg.norm(step) <= cfg.tol * (1.0 + np.linalg.norm(w_new)):
            w = w_new
            break
        w = w_new
    return w


def run_channel_equalization_logistic(
    observed: np.ndarray,
    target: np.ndarray,
    washout: int,
    train_len: int,
    test_len: int,
    logistic_cfg: LogisticEqualizerConfig,
    metric: Literal["ber", "mse"] = "ber",
) -> Dict[str, np.ndarray | float]:
    observed = np.asarray(observed, dtype=float).reshape(-1)
    target = np.asarray(target, dtype=float).reshape(-1)
    if observed.shape[0] != target.shape[0]:
        raise ValueError("observed and target must have the same length.")
    if metric not in {"ber", "mse"}:
        raise ValueError("metric must be 'ber' or 'mse'.")

    X = _lagged_design_matrix(observed, logistic_cfg.n_lags)
    t0 = int(max(washout, logistic_cfg.n_lags - 1))
    t_train_end = t0 + int(train_len)
    t_test_end = t_train_end + int(test_len)
    tr = np.arange(t0, t_train_end)
    te = np.arange(t_train_end, t_test_end)

    y01 = ((target + 1.0) * 0.5).astype(float)
    w = _fit_logistic_regression(X[tr], y01[tr], logistic_cfg)

    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        logits_tr = X[tr] @ w
        logits_te = X[te] @ w
    if not np.isfinite(logits_tr).all() or not np.isfinite(logits_te).all():
        raise FloatingPointError("Logistic regression predictions became non-finite.")
    prob_tr = _sigmoid(logits_tr)
    prob_te = _sigmoid(logits_te)
    yhat_tr = 2.0 * prob_tr - 1.0
    yhat_te = 2.0 * prob_te - 1.0
    pred_tr = np.where(prob_tr >= 0.5, 1.0, -1.0)
    pred_te = np.where(prob_te >= 0.5, 1.0, -1.0)

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
        "train_pred": pred_tr,
        "test_pred": pred_te,
        "train_signal": yhat_tr,
        "test_signal": yhat_te,
        "weights": w,
    }


def run_channel_equalization_symbol_logistic(
    train_observed: np.ndarray,
    train_messages: np.ndarray,
    test_observed: np.ndarray,
    test_messages: np.ndarray,
    readout_cfg: SoftmaxReadoutConfig,
    n_lags: int = 1,
) -> Dict[str, Any]:
    X_train = _message_lagged_design(train_observed, n_lags)[:, :, 1:].reshape(-1, n_lags)
    X_test = _message_lagged_design(test_observed, n_lags)[:, :, 1:].reshape(-1, n_lags)
    y_train = np.asarray(train_messages).reshape(-1)
    y_test = np.asarray(test_messages).reshape(-1)

    model = fit_softmax_readout(X_train, y_train, readout_cfg)
    train_pred = predict_softmax_readout(X_train, model)
    test_pred = predict_softmax_readout(X_test, model)

    train_error = float(np.mean(train_pred != y_train))
    test_error = float(np.mean(test_pred != y_test))
    return {
        "train_error_rate": train_error,
        "test_error_rate": test_error,
        "train_pred": train_pred.reshape(np.asarray(train_messages).shape),
        "test_pred": test_pred.reshape(np.asarray(test_messages).shape),
        "model": model,
    }
