"""Forecast metrics with strict alignment and undefined-value semantics."""
import numpy as np


def _pair(y_true, y_pred):
    a, b = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or not a.size:
        raise ValueError("expected aligned nonempty one-dimensional vectors")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite observations")
    return a, b


def mae(y_true, y_pred):
    a, b = _pair(y_true, y_pred)
    return float(np.mean(np.abs(a - b)))


def mse(y_true, y_pred):
    a, b = _pair(y_true, y_pred)
    return float(np.mean((a - b) ** 2))


def mape(y_true, y_pred):
    a, b = _pair(y_true, y_pred)
    if np.any(a == 0):
        raise ValueError("MAPE is undefined at zero actual values")
    return float(100 * np.mean(np.abs((a - b) / a)))
