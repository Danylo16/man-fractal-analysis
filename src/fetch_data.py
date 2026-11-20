"""Download monthly index observations locally; review upstream data rights."""
import argparse
from pathlib import Path
import pandas as pd

URL = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("sp500_local.csv"))
    a = p.parse_args()
    d = pd.read_csv(URL)
    d = d.loc[d.Date.between("2000-01-01", "2024-12-01"), ["Date", "SP500"]]
    if len(d) != 300:
        raise ValueError("unexpected source range")
    with a.out.open("x", encoding="utf-8", newline="") as f:
        d.to_csv(f, index=False)
