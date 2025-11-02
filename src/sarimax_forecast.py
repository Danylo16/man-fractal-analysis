# src/sarimax_forecast.py
import argparse
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from metrics import mae, mse, mape

def fit_forecast(y: np.ndarray, exog=None, order=(1,0,1), seasonal=(0,0,0,0), steps=30):
    model = sm.tsa.SARIMAX(
        y, exog=exog, order=order,
        seasonal_order=seasonal, enforce_stationarity=False, enforce_invertibility=False
    )
    res = model.fit(disp=False)
    fc = res.get_forecast(steps=steps, exog=exog[-steps:] if exog is not None and len(exog) >= steps else None)
    mean = fc.predicted_mean
    conf = fc.conf_int(alpha=0.05)
    return res, mean, conf

def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with time series")
    ap.add_argument("--y", required=True, help="column name for target")
    ap.add_argument("--exo", default="", help="comma-separated exogenous columns")
    ap.add_argument("--order", default="1,0,1", help="p,d,q")
    ap.add_argument("--seasonal", default="0,0,0,0", help="P,D,Q,s")
    ap.add_argument("--h", type=int, default=30)
    ap.add_argument("--metrics_out", default="", help="optional path to metrics json if y contains last h true values")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    y = df[args.y].astype(float).to_numpy()
    exo_cols = [c for c in args.exo.split(",") if c]
    exog = df[exo_cols].to_numpy() if exo_cols else None

    order = tuple(int(x) for x in args.order.split(","))
    seasonal = tuple(int(x) for x in args.seasonal.split(","))
    res, mean, conf = fit_forecast(y, exog=exog, order=order, seasonal=seasonal, steps=args.h)

    print(res.summary())
    print("\nForecast head:")
    print(mean[:5])

    # якщо у файлі останні h рядків — це «майбутнє з відомими y», можна порахувати метрики
    if args.metrics_out:
        y_true = y[-args.h:] if len(y) >= args.h else None
        if y_true is not None:
            m = {"mae": mae(y_true, mean[:args.h]), "mse": mse(y_true, mean[:args.h]), "mape": mape(y_true, mean[:args.h])}
            with open(args.metrics_out, "w", encoding="utf-8") as f:
                json.dump(m, f, indent=2)
            print("Saved metrics ->", args.metrics_out)

if __name__ == "__main__":
    _cli()
