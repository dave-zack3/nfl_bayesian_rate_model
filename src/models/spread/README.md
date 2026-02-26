# Spread Model (Dynamic AR(1))

This model directly predicts game-level point spread using a
Bayesian dynamic state-space framework.

## Structure

Latent team strength evolves as:

    theta_t = rho * theta_{t-1} + eps_t

Where:
- rho ∈ (0,1) controls persistence
- eps_t ~ Normal(0, sigma_theta)

At the start of each season:

    theta_t += eta_season

Where:
- eta_season ~ Normal(0, sigma_offseason)

This captures offseason roster / coaching / structural changes.

## Observation Model

Spread = theta_home − theta_away + beta_home + noise

Noise options:
- Homoskedastic Normal
- Heteroskedastic Normal (variance increases with expected dominance)

## Interpretable Parameters

- rho: strength persistence across weeks
- sigma_theta: weekly volatility
- sigma_offseason: offseason structural change magnitude
- beta_home: home-field advantage
- avg_abs_strength: league parity measure
- offseason_shock_magnitude: average offseason movement

## Motivation

The model improves on Elo by:

- Explicitly modeling strength evolution
- Allowing structural breaks
- Modeling uncertainty probabilistically