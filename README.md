# Dynamic Bayesian NFL Scoring Model

## Overview

This project implements a hierarchical Bayesian Negative Binomial model for NFL team scoring with:

- Dynamic offense and defense effects (Gaussian Random Walk)
- Home field advantage
- Drive-adjusted scoring rate
- Negative Binomial likelihood (overdispersion)
- Rolling expanding-window evaluation
- Betting market comparison
- Elo benchmark comparison

The model is built in **PyMC v5** and evaluated via:

- RMSE / MAE (spread prediction)
- Brier score (win probability calibration)
- Log Predictive Density
- Betting edge vs Vegas
- Sharpness (posterior variance)
- Elo benchmark comparison

---

## Model Structure

For team *i* vs opponent *j* in week *t*:

log(λᵢₜ) =
    intercept
    + offenseᵢₜ
    − defenseⱼₜ
    + home_effectᵢ
    + β_pace * log(drives)

Points ~ NegativeBinomial(μ = exp(log(λ)), α = φ)

Offense and defense follow Gaussian Random Walks across weeks.

---

## Key Features

- Dynamic team strength evolution
- Hierarchical shrinkage
- Proper Bayesian missing-data forecasting
- Rolling out-of-sample evaluation
- Automated experiment logging
- Elo benchmark comparison

---

## Running the Pipeline

```bash
python -m src.run_pipeline