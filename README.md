# Statistical Analysis of Electricity Consumption Using Fractal Methods

Research project by Danylo Zubrii, Junior Academy of Sciences of Ukraine, Mathematics / Statistics section (2024). This repository is undergoing a 2026 reproducibility revision.


## Research documents

- [English translation (PDF)](docs/original-research-2024/MAN_Research_Paper_EN.pdf) — AI-assisted translation prepared in 2026.
- [Original Ukrainian manuscript (DOCX)](docs/original-research-2024/Робота_Зубрій_Данило_Владиславович.docx) — 2024.
- [Errata and methodological notes](docs/original-research-2024/ERRATA.md).

## Original study (2024)

The study examined 1,553 daily electricity-consumption observations from Rivne region, Ukraine (2019-07-01 through 2023-09-30), supplied by Rivneoblenergo. It investigated seasonal structure, temporal dependence and associations with daily temperature, comparing statistical and machine-learning forecasting approaches.

Methods reported in the original manuscript: rescaled-range (R/S) analysis, Hurst exponent estimation, spectral analysis, seasonal decomposition, ACF/PACF, Dickey–Fuller testing, AIC-based model selection, Random Forest feature importance, SARIMAX and XGBoost.

| Historical reported statistic | Value |
| --- | ---: |
| Hurst exponent, original series | 0.7259 |
| Hurst exponent, shuffled series | 0.5468 |
| Pearson correlation with temperature | -0.802 |

These are historical manuscript results, not independently reproduced measurements. A Hurst estimate above 0.5 alone does not establish long memory, particularly in a seasonal or nonstationary series. The original manuscript's fractal-dimension arithmetic also requires correction: 2 - 0.7259 = 1.2741, subject to the self-affinity assumptions of that relationship.

## Repository status

The legacy Python modules implement R/S estimation, periodogram-based cycle detection, SARIMAX forecasting and error metrics. The original study included additional methods that are not yet fully restored here.

The 2026 revision adds a separate exploratory study of monthly S&P 500 index averages, chronological forecast evaluation, explicit future covariates and numerical tests. See [S&P 500 study and reproduction instructions](README_RESEARCH_2026.md) and [revision notes](docs/revision-2026.md).

## Data availability

The original utility dataset is not distributed here. Its availability and redistribution permission have not been established. The legacy sample CSV is a placeholder, not a research dataset. Full reproduction of the 2024 results requires the original observations and experimental configuration.

## Development

Use a virtual environment and install dependencies:

    python -m pip install -r requirements-research.txt

Use the research-specific requirements and commands in the linked S&P 500 study. The original electricity dataset is unavailable; the new study is not a replication of the 2024 results.

## Research integrity

The 2024 study and the 2026 revision are distinct. Corrections and new experiments will be documented as such, rather than retroactively presented as original results.
