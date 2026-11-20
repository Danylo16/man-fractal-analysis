# S&P 500: fractal and spectral research extension (2026)

AI-assisted exploratory extension of the 2024 Junior Academy of Sciences electricity-demand research. Original utility data are unavailable; historical results are not reproduced.

## Reproduce

Python 3.11+; from repository root:

    python -m pip install -r requirements-research.txt
    python -m src.fetch_data --out sp500_local.csv
    python -m pytest -q
    python -m src.research sp500_local.csv --out results

Source: https://github.com/datasets/s-and-p-500/blob/main/data/data.csv (Robert Shiller / FRED). January 2000–December 2024, 300 monthly average index observations, not month-end prices or total returns. Raw data are not committed; check source rights before redistribution. Upstream revisions may change results.

## Method
Nonoverlapping R/S slopes at 8, 12, 16, 24, 32, 48 and 64 months (at least four blocks); 499 fixed-seed permutations per series; ADF and Hann-window spectrum; sensitivity to linear detrending, monthly seasonal means and AR(1) residualization. Expanding one-step evaluation over 60 monthly log changes with fixed zero, historical mean, seasonal naive, AR(1) and XGBoost models.

## Local exploratory results
Reported R/S slopes: log price 1.0420, log change 0.7289, absolute log change 0.7772. Raw level slope above one is not a conventional Hurst exponent. Permutation tests destroy short- and long-range dependence and do not establish long memory. Monthly aggregation can induce dependence. Multiple tests unadjusted.

Forecast RMSE in percentage log-change units: XGBoost 4.2594, historical mean 4.3187, AR(1) 4.3698, zero 4.3915, seasonal naive 5.9649. The small XGBoost advantage is not established as statistically significant or tradeable. These are previous local experiment results, not independently replicated by GitHub CI.

Notebooks provide synthetic-method demonstrations and an optional local S&P 500 workflow; original energy experiments remain incomplete. Future work: daily data, dependence-preserving surrogate tests, DFA, confidence intervals and independent holdout. No trading simulation is performed here.
