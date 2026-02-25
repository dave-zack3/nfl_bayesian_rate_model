# NFL Bayesian Team Strength & Spread Modeling

A research-driven Bayesian framework for estimating team strength and predicting NFL point spreads, evaluated against an Elo benchmark using rolling out-of-sample backtesting.

---

## Objective

1. Estimate offensive and defensive team strength relative to league average.
2. Model temporal evolution of team strength.
3. Predict game spreads.
4. Benchmark performance against Elo.
5. Improve upon Elo using principled probabilistic modeling.

---

## Model Overview

### 1. Generative Scoring Model
- Hierarchical offense & defense parameters
- Negative Binomial likelihood (handles over-dispersion)
- AR(1) strength evolution:
  
  \[
  \theta_t = \rho \theta_{t-1} + \epsilon_t
  \]

- Sum-to-zero constraint each week for identifiability

Outputs:
- Posterior strength trajectories
- Persistence parameter (ρ)
- Dispersion parameter (φ)

---

### 2. Direct Spread Model

Rather than deriving spread from simulated scores, a separate Gaussian spread model is estimated:

\[
Spread \sim \mathcal{N}(\theta_{home} - \theta_{away} + \beta_{home}, \sigma)
\]

#### MOV Heteroskedastic Extension

Residual variance scales with expected dominance:

\[
\sigma_i = \sigma_0 (1 + \alpha |\mu_i|)
\]

This mimics Elo’s margin-of-victory adjustment in a fully generative Bayesian framework.

---

## Evaluation Framework

- Rolling expanding-window cross-validation
- Final season held out for testing
- Metrics:
  - RMSE (primary)
  - MAE
  - Log Predictive Density
  - Predictive sharpness

All comparisons are strictly out-of-sample.

---

## Benchmark

A standard Elo model is implemented for reference.

Purpose:
- Provide baseline performance
- Identify structural advantages/disadvantages
- Prevent overfitting illusions

---

## Key Findings

- Poisson under-dispersed scoring relative to observed variance.
- Negative Binomial improved posterior predictive alignment.
- AR(1) bounded variance relative to random walk.
- Direct spread modeling reduced variance inflation from generative scoring.
- MOV heteroskedastic scaling introduced adaptive uncertainty similar to Elo.

---

## Repository Structure

- `model/` – Generative scoring model
- `spread_model/` – Direct spread model
- `evaluation/` – Rolling backtest + Elo benchmark
- `data/` – Multi-season processing pipeline
- `logs/` – Reproducible experiment records

---

## Why This Project Matters

This project demonstrates:

- Hierarchical Bayesian modeling
- Time-series state evolution (AR(1))
- Proper posterior predictive checking
- Out-of-sample rolling validation
- Benchmark comparison
- Controlled model iteration

The design prioritizes interpretability, statistical discipline, and reproducibility over feature sprawl.

---

## Next Steps

- Structured margin-of-victory update dynamics
- Time-varying volatility
- Prior calibration experiments
- Further benchmarking

---

## Author Note

All modeling decisions were stress-tested via posterior predictive checks and out-of-sample evaluation.  
Complexity was added only when justified by performance or statistical necessity.