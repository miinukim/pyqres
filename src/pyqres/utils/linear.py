from __future__ import annotations
import numpy as np

def ridge_regression_fit(X: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    if y.ndim == 1:
        y2 = y[:, None]
    else:
        y2 = y
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise FloatingPointError("Non-finite values in regression inputs (X or y).")
    F = X.shape[1]
    # Keep regression robust even when global np.seterr(all="raise") is active.
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        A = X.T @ X + l2 * np.eye(F)
        B = X.T @ y2
    if not np.isfinite(A).all() or not np.isfinite(B).all():
        raise FloatingPointError("Non-finite values in normal equations (A or B).")
    w = np.linalg.solve(A, B)
    return w.squeeze()

def ridge_regression_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    if not np.isfinite(X).all() or not np.isfinite(w).all():
        raise FloatingPointError("Non-finite values in prediction inputs (X or w).")
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        yhat = X @ w
    if not np.isfinite(yhat).all():
        raise FloatingPointError("Non-finite values in predictions yhat.")
    return yhat

def rmse(yhat: np.ndarray, y: np.ndarray) -> float:
    yhat = np.asarray(yhat); y = np.asarray(y)
    return float(np.sqrt(np.mean((yhat - y) ** 2)))

def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y).reshape(-1); yhat = np.asarray(yhat).reshape(-1)
    denom = np.var(y)
    if denom <= 1e-12:
        return 0.0
    return float(1.0 - np.mean((y - yhat) ** 2) / denom)
