# Points Model (Generative Scoring Model)

This module estimates team offensive and defensive strength via a Bayesian hierarchical framework.

---

## Likelihood

Originally:
- Poisson scoring model (mean = variance)

Upgraded to:
- Negative Binomial
- Var(Y) = μ + μ² / φ
- Allows over-dispersion relative to Poisson

---

## Dynamic Evolution

### Random Walk
θ_t = θ_{t-1} + ε_t  
- Unbounded variance over time

### AR(1) (Current)
θ_t = ρ θ_{t-1} + ε_t  
- Bounded stationary variance
- Mean reversion controlled by ρ
- Global persistence parameter

---

## Identifiability

- Sum-to-zero constraint each week
- Global ρ to avoid overfitting
- Shared innovation variance

---

## Structural Outputs

- Posterior mean strength
- Credible intervals
- ρ persistence estimate
- Innovation variance
- Dispersion parameter φ