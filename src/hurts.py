import numpy as np
import argparse
import pandas as pd

def hurst_rs(x: np.ndarray, min_win: int = 16, max_win: int | None = None, n_scales: int = 8):
    """
    Estimate Hurst exponent using classical R/S analysis.
    Returns (H, details) where details contains scales and log(R/S) values.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if max_win is None:
        max_win = n // 4
    if max_win <= min_win:
        raise ValueError("Series too short for R/S")

    # Geometrically spaced scales
    scales = np.unique(
        np.round(np.geomspace(min_win, max_win, n_scales)).astype(int)
    )
    log_rs = []
    for m in scales:
        k = n // m
        if k < 2:
            continue
        seg = x[:k*m].reshape(k, m)
        # Centered cumulative deviations
        y = seg - seg.mean(axis=1, keepdims=True)
        z = np.cumsum(y, axis=1)
        R = z.max(axis=1) - z.min(axis=1)
        S = y.std(axis=1, ddof=1)
        rs = np.where(S > 0, R / S, np.nan)
        rs = rs[~np.isnan(rs)]
        if rs.size:
            log_rs.append((np.log(m), np.log(np.mean(rs))))
    if len(log_rs) < 2:
        raise ValueError("Insufficient valid R/S points")

    X = np.array([p[0] for p in log_rs])
    Y = np.array([p[1] for p in log_rs])
    A = np.vstack([X, np.ones_like(X)]).T
    H, c = np.linalg.lstsq(A, Y, rcond=None)[0]
    details = {"scales": X, "log_rs": Y, "intercept": c}
    return float(H), details

def fractal_dimension_from_hurst(H: float) -> float:
    return 2.0 - float(H)

def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to CSV with a column")
    ap.add_argument("--col", required=True, help="column name to analyze")
    ap.add_argument("--min_win", type=int, default=16)
    ap.add_argument("--max_win", type=int, default=None)
    ap.add_argument("--n_scales", type=int, default=8)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    x = df[args.col].to_numpy()
    H, det = hurst_rs(x, args.min_win, args.max_win, args.n_scales)
    D = fractal_dimension_from_hurst(H)
    print(f"H={H:.3f}  D=2−H={D:.3f}")
    for a, b in zip(det["scales"], det["log_rs"]):
        print(f"log(m)={a:.3f}  log(R/S)={b:.3f}")

if __name__ == "__main__":
    _cli()
