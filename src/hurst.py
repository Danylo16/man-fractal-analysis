"""Classical non-overlapping rescaled-range estimator."""
import numpy as np


def hurst_rs(x, min_win=16, max_win=None, n_scales=8, *, scales=None, min_blocks=2):
    """Return the log-log slope and diagnostics, not a long-memory proof."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("expected a finite one-dimensional series")
    n = len(x)
    max_win = n // 4 if max_win is None else max_win
    if min_blocks < 2:
        raise ValueError("at least two blocks required")
    if scales is None:
        if not 2 <= min_win < max_win <= n // 2 or n_scales < 2:
            raise ValueError("invalid scale range")
        scales = np.unique(np.round(np.geomspace(min_win, max_win, n_scales)).astype(int))
    else:
        scales = sorted(set(scales))
        if any(not isinstance(m, (int, np.integer)) or m < 2 for m in scales):
            raise ValueError("invalid explicit scale")
    valid, values, counts = [], [], []
    for m in scales:
        if n // m < min_blocks:
            continue
        blocks = x[: (n // m) * m].reshape(-1, m)
        centered = blocks - blocks.mean(axis=1, keepdims=True)
        cumulative = np.cumsum(centered, axis=1)
        ranges = np.ptp(np.column_stack((np.zeros(len(blocks)), cumulative)), axis=1)
        std = centered.std(axis=1, ddof=1)
        mask = (std > 0) & (ranges > 0)
        if mask.any():
            rs = np.mean(ranges[mask] / std[mask])
            if np.isfinite(rs) and rs > 0:
                valid.append(m)
                values.append(rs)
                counts.append(int(mask.sum()))
    if len(valid) < 2:
        raise ValueError("insufficient valid R/S scales")
    xx, yy = np.log(valid), np.log(values)
    h, intercept = np.linalg.lstsq(np.column_stack((xx, np.ones_like(xx))), yy, rcond=None)[0]
    residual = yy - (h * xx + intercept)
    total = np.sum((yy - yy.mean()) ** 2)
    return float(h), {"scales": xx, "log_rs": yy, "intercept": float(intercept),
                      "window_sizes": np.asarray(valid), "block_counts": counts,
                      "r_squared": float(1 - np.sum(residual ** 2) / total) if total > 0 else None}


def fractal_dimension_from_hurst(h):
    """Only for a graph satisfying the appropriate self-affinity assumptions."""
    if not np.isfinite(h) or not 0 < h < 1:
        raise ValueError("H must lie strictly between zero and one")
    return 2.0 - float(h)
