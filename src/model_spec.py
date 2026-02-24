import numpy as np
import pymc as pm

def build_model(team_idx, opp_idx, time_idx,
                points, drives, home_flag,
                n_teams, n_weeks):

    with pm.Model() as model:

        # ----------------------------------------
        # Random walk innovation scales
        # ----------------------------------------
        # Controls how much team strength evolves week-to-week
        sigma_off_rw = pm.HalfNormal("sigma_off_rw", 0.3)
        sigma_def_rw = pm.HalfNormal("sigma_def_rw", 0.3)

        # ----------------------------------------
        # Dynamic offense latent states
        # Shape: (teams, weeks)
        # init_dist anchors week 0 to avoid diffuse geometry
        # ----------------------------------------
        offense_raw = pm.GaussianRandomWalk(
            "offense_raw",
            sigma=sigma_off_rw,
            shape=(n_teams, n_weeks),
            init_dist=pm.Normal.dist(0, 0.3)
        )

        # ----------------------------------------
        # Dynamic defense latent states
        # ----------------------------------------
        defense_raw = pm.GaussianRandomWalk(
            "defense_raw",
            sigma=sigma_def_rw,
            shape=(n_teams, n_weeks),
            init_dist=pm.Normal.dist(0, 0.3)
        )

        # ----------------------------------------
        # Weekly sum-to-zero constraint
        # Ensures identifiability of team strengths
        # ----------------------------------------
        offense = offense_raw - pm.math.mean(
            offense_raw, axis=0, keepdims=True
        )

        defense = defense_raw - pm.math.mean(
            defense_raw, axis=0, keepdims=True
        )

        # ----------------------------------------
        # Structural parameters
        # ----------------------------------------
        intercept = pm.Normal("intercept", 0, 1)

        sigma_home = pm.HalfNormal("sigma_home", 0.5)
        home_raw = pm.Normal("home_raw", 0, 1, shape=n_teams)
        home_effect = home_raw * sigma_home

        beta_pace = pm.Normal("beta_pace", 1.0, 0.3)

        # NB dispersion
        phi = pm.HalfNormal("phi", 5)

        # ----------------------------------------
        # Linear predictor
        # ----------------------------------------
        log_lambda = (
            intercept
            + offense[team_idx, time_idx]
            - defense[opp_idx, time_idx]
            + home_effect[team_idx] * home_flag
            + beta_pace * np.log(drives)
        )

        # ----------------------------------------
        # Likelihood
        # If points contains NaN values,
        # PyMC automatically treats them as missing
        # and generates posterior predictive samples
        # ----------------------------------------
        pm.NegativeBinomial(
            "points",
            mu=pm.math.exp(log_lambda),
            alpha=phi,
            observed=points
        )

    return model