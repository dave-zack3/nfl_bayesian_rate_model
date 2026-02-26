# Evaluation Framework

We use expanding-window backtesting:

For each week:
- Fit on all prior data
- Mask current week outcomes
- Generate posterior predictive distribution
- Compute metrics

## Metrics

- RMSE
- MAE
- Log Predictive Density (LPD)
- Sharpness
- Brier score (for win probability)
- Betting edge (optional)

## Why LPD Matters

RMSE measures point accuracy.
LPD measures probabilistic calibration.

We prioritize LPD when comparing Bayesian models.