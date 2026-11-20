# 2026 reproducibility revision

## Priority defects

- Forecast CLI fits the complete series and compares future forecasts to historical observations. Replace with an explicit chronological holdout.
- Historical exogenous values are reused as future covariates. Require separately supplied future covariates or clearly label conditional forecasts.
- Notebook imports src.hurst but implementation is src/hurts.py.
- Placeholder sample CSV cannot support reproducible numerical experiments.
- R/S inference needs scale diagnostics, finite-input checks and repeated permutation controls. Trend and seasonality can mimic long memory.
- Periodogram uses a Hann window and spectrum scaling; document this accurately rather than describing output as power spectral density.
- MAPE near zero needs an explicit undefined-value policy.
- Historical manuscript reports impossible p-values in its Dickey–Fuller table and an arithmetic error in fractal dimension. Do not infer missing scientific notation without original outputs.

## Acceptance criteria

1. Deterministic synthetic fixtures with known seasonal periods; clearly labeled as synthetic.
2. Tests for constant and invalid series, known forecast alignment and exogenous dimensions.
3. Seasonal-naive baseline and chronological holdout evaluation.
4. Repeated permutation experiment, detrended/deseasonalized sensitivity analysis, and explicit assumptions.
5. Separate original reported results from newly executed experiments.
6. Restore the full original methodology where data and source materials permit; document remaining limitations.

No 2026 experiments have been executed as of this documentation commit.
