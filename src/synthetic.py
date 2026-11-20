"""Generate a labeled synthetic daily workload; not original utility data."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(days=1553, seed=42):
    if days < 1:
        raise ValueError("days must be positive")
    rng = np.random.default_rng(seed)
    t = np.arange(days)
    temperature = 10 + 12 * np.sin(2 * np.pi * t / 365) + rng.normal(0, 2, days)
    value = 100 - 1.5 * temperature + 5 * np.cos(2 * np.pi * t / 7) + rng.normal(0, 2, days)
    return pd.DataFrame({"date": pd.date_range("2019-07-01", periods=days),
                         "value": value, "temperature": temperature})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--days", type=int, default=1553)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="") as handle:
        generate(args.days, args.seed).to_csv(handle, index=False)


if __name__ == "__main__":
    main()
