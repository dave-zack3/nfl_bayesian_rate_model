# Evaluation Framework

This module provides rigorous out-of-sample testing.

---

## Rolling Expanding Window

For each week in the test season:

1. Train on all prior data
2. Forecast next week
3. Record metrics

Prevents look-ahead bias.

---

## Metrics

- RMSE (primary comparison metric)
- MAE
- Log Predictive Density
- Sharpness (predictive variance)

---

## Elo Benchmark

Standard Elo model implemented for comparison.

Purpose:
- Provide simple baseline
- Identify structural gaps
- Prevent self-referential validation

---

## Philosophy

Performance must be demonstrated out-of-sample.

Structural diagnostics alone are insufficient.