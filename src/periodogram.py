# src/periodogram.py
import numpy as np
import pandas as pd
import argparse
from scipy.signal import periodogram, find_peaks

def schuster_periodogram(x: np.ndarray, fs: float):
    """
    Returns frequencies (Hz) and power spectral density via classical periodogram.
    """
    f, Pxx = periodogram(x, fs=fs, window="hann", scaling="spectrum", detrend="constant")
    return f, Pxx

def top_periods(x: np.ndarray, fs: float, top_k: int = 3, min_prom: float = 0.0):
    f, P = schuster_periodogram(x, fs)
    # ігноруємо 0 Гц
    f, P = f[1:], P[1:]
    peaks, props = find_peaks(P, prominence=min_prom)
    order = np.argsort(P[peaks])[::-1][:top_k]
    peaks = peaks[order]
    freqs = f[peaks]
    periods = 1.0 / freqs
    powers = P[peaks]
    return periods, freqs, powers, (f, P)

def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--col", required=True)
    ap.add_argument("--fs", type=float, required=True, help="sampling rate (Hz)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--prom", type=float, default=0.0)
    args = ap.parse_args()

    x = pd.read_csv(args.csv)[args.col].to_numpy(dtype=float)
    periods, freqs, powers, _ = top_periods(x, args.fs, args.k, args.prom)
    for i, (T, f, p) in enumerate(zip(periods, freqs, powers), 1):
        print(f"#{i}: period={T:.3f}s  freq={f:.4f}Hz  power={p:.3e}")

if __name__ == "__main__":
    _cli()
