# Direct Spread Model

This model predicts game spread directly rather than deriving it from scoring distributions.

---

## Baseline Model

Spread ~ Normal(μ, σ)

μ = θ_home - θ_away + β_home

Where θ follows:

- Static (season-level)
- AR(1) dynamic evolution

---

## MOV Heteroskedastic Extension

Residual variance scales with expected dominance:

σ_i = σ₀ (1 + α |μ_i|)

This mimics Elo’s margin-of-victory adjustment in a fully generative Bayesian framework.

---

## Why Direct Spread?

The generative points model can inflate predictive spread variance due to:
- Independent scoring noise
- NB over-dispersion

Direct modeling aligns likelihood with evaluation metric (spread RMSE).

---

## Interpretation

- α > 0 implies larger mismatches have higher uncertainty.
- If α ≈ 0, homoskedastic assumption suffices.