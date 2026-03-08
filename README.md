# NFL Bayesian Team Strength & Spread Modeling

A research-driven Bayesian framework for estimating dynamic NFL team strength and predicting point spreads, evaluated against an Elo benchmark using rolling out-of-sample backtesting.

---

## Objective

This project builds a probabilistic alternative to Elo for modeling NFL team strength and game spreads.

Goals:

1. Estimate dynamic team strength.
2. Model week-to-week persistence and offseason shifts.
3. Predict point spreads with calibrated uncertainty.
4. Compare performance against a standard Elo benchmark.
5. Evaluate models strictly out-of-sample using rolling validation.

---

## Model Architecture

### Dynamic Team Strength Model

Team strength evolves according to an AR(1) process:


```math
\theta_t = \rho \theta_{t-1} + \epsilon_t
```

Where:

- $\( \rho \)$ = persistence parameter  
- \( \epsilon_t \sim \mathcal{N}(0, \sigma_{weekly}) \)

An offseason volatility term allows structural shifts between seasons:

\[
\theta_{start\ of\ season} += \eta_s
\]

Where:

- \( \eta_s \sim \mathcal{N}(0, \sigma_{offseason}) \)

A sum-to-zero constraint ensures identifiability each week.

---

### Direct Bayesian Spread Model

Rather than simulating scores, spreads are modeled directly:

\[
Spread \sim \mathcal{N}(
\theta_{home} - \theta_{away} + \beta_{home},
\sigma_i
)
\]

Two noise specifications are evaluated:

- **Homoskedastic**
- **Heteroskedastic (Margin-of-Victory Scaling)**

Heteroskedastic form:

\[
\sigma_i = \sigma_0 (1 + \alpha |\mu_i|)
\]

This mimics Elo’s margin-of-victory adjustment in a fully probabilistic framework.

---

## Evaluation Framework

All performance is evaluated using:

- Rolling expanding-window backtesting
- Final season held out for testing
- Strictly no data leakage

Metrics:

- RMSE
- MAE
- Log Predictive Density
- Predictive Sharpness

---

## Benchmark

A standard Elo model is implemented for reference.

Purpose:

- Provide baseline comparison
- Ensure improvements are meaningful
- Avoid overfitting illusions

---

## Experimental Design (Whitepaper Set)

The final whitepaper compares:

1. **Bayesian (Heteroskedastic)**
2. **Bayesian (Homoskedastic)**
3. **Elo Baseline**

Structural fits are sampled with:

- 500 draws
- 500 tuning iterations
- NUTS sampler (target_accept=0.95)

All structural traces are saved to NetCDF for reproducibility.

---

## Repository Structure

data/
    processed datasets

src/
    data_loader.py
    models/
        evaluation/
            fit_model.py
            run_pipeline.py

whitepaper_outputs/
    saved traces (.nc)
    rolling backtest CSVs

---

## Key Structural Findings

- Team strength is highly persistent (ρ ≈ 0.98).
- Weekly volatility exceeds offseason volatility.
- Home-field advantage ≈ 1.6 points.
- Heteroskedastic scaling improves predictive calibration.

---

## Why This Matters

This project demonstrates:

- Hierarchical Bayesian modeling
- Dynamic state-space modeling
- Probabilistic forecasting
- Strict out-of-sample validation
- Benchmark-driven evaluation
- Reproducible experiment design

It prioritizes interpretability, statistical discipline, and transparent evaluation over feature sprawl.

---

## Next Steps

- Time-varying volatility
- Injury-adjusted priors
- Play-level feature augmentation
- Betting edge analysis
- Expanded multi-league evaluation

---

## Author

Developed as a portfolio sports analytics research project focused on interpretable probabilistic modeling and rigorous evaluation.
