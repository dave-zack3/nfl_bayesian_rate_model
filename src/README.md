# Source Code Overview

This directory contains all modeling, evaluation, and experiment orchestration logic.

---

## Data Pipeline

### data_loader.py

- Downloads NFL play-by-play data via `nfl_data_py`
- Constructs team-game level dataset
- Builds global time index across seasons
- Creates game-level spread dataset
- Saves metadata JSON

---

## Spread Model

### spread_model_spec.py

Defines the dynamic Bayesian spread model:

- AR(1) team strength
- Offseason volatility term
- Sum-to-zero constraint per week
- Home-field parameter
- Configurable noise structure:
  - homoskedastic
  - heteroskedastic (MOV scaling)

Key parameters:

- rho
- sigma_theta (weekly volatility)
- sigma_offseason
- beta_home
- sigma0
- alpha (heteroskedastic only)

---

### spread_model_wrapper.py

Builds PyMC model from dataframe inputs:

- team indices
- time index
- season id
- season start indicator
- spread
- home flag

Handles tensor formatting and shape alignment.

---

### spread/diagnostics.py

`summarize_spread_structure(trace)` extracts:

- Posterior mean
- Standard deviation
- 94% HDI
- Average absolute strength magnitude

Used for structural reporting.

---

## Backtesting

### rolling_backtest_spread.py

Implements:

- Expanding window training
- Weekly test prediction
- Posterior predictive sampling
- Metric computation

Ensures:

- No leakage
- Clean separation of train/test
- Predictive metrics computed only on held-out games

---

## Metrics

metrics.py provides:

- RMSE
- MAE
- Brier score
- Log predictive density
- Calibration curve
- Betting edge
- Sharpness

All metrics are vectorized and evaluation-safe.

---

## Experiment Pipeline

### run_pipeline.py

Coordinates:

1. Dataset construction
2. Structural fit (500/500 draws)
3. Trace saving (NetCDF)
4. Rolling backtest
5. Elo benchmark
6. Experiment logging

Artifacts saved to:

whitepaper_outputs/

- *_trace.nc
- *_rolling.csv

---

## Reproducibility

- All experiments logged via experiment_logger.py
- Structural traces saved
- Rolling results persisted
- Deterministic train/test splitting
- Fixed random seeds where applicable

---

## Whitepaper Integration

Saved trace files can be loaded via:

```python
import arviz as az
trace = az.from_netcdf("whitepaper_outputs/Bayesian_Heteroskedastic_trace.nc")

This allows figure and table generation without re-fitting models.