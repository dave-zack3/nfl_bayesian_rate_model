# Evaluation Framework

This directory contains all out-of-sample evaluation logic.

---

## rolling_backtest.py

Implements expanding-window rolling evaluation.

For each week:

- Train on all previous data
- Mask current week points
- Sample posterior
- Generate posterior predictive
- Compute spread metrics

---

## spread_metrics.py

Implements:

- RMSE
- MAE
- Brier score
- Log Predictive Density
- Betting edge
- Sharpness

---

## elo_model.py

Baseline Elo rating model.

- Initialized at 1500
- Logistic expectation
- K-factor update
- Home advantage adjustment

---

## elo_metrics.py

Computes benchmark RMSE, MAE, and Brier score.

---

## vegas_loader.py

Loads and formats Vegas closing spread data.