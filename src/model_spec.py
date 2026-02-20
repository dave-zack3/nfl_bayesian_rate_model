import numpy as np
import pymc as pm

def build_model(team_idx, opp_idx, points, drives, home_flag, n_teams):

    with pm.Model() as model:

        # Hyperpriors
        sigma_off = pm.HalfNormal("sigma_off", 1.0)
        sigma_def = pm.HalfNormal("sigma_def", 1.0)

        # Team effects
        offense_raw = pm.Normal("offense_raw", 0, 1, shape=n_teams)
        defense_raw = pm.Normal("defense_raw", 0, 1, shape=n_teams)

        offense = offense_raw * sigma_off
        defense = defense_raw * sigma_def

        # Sum-to-zero constraint
        offense = offense - pm.math.mean(offense)
        defense = defense - pm.math.mean(defense)

        # Intercept + home field
        intercept = pm.Normal("intercept", 0, 1)
        beta_home = pm.Normal("beta_home", 0, 1)

        # Linear predictor
        log_lambda = (
            intercept
            + offense[team_idx]
            - defense[opp_idx]
            + beta_home * home_flag
            + np.log(drives)
        )

        # Likelihood
        pm.Poisson("points_obs", mu=pm.math.exp(log_lambda), observed=points)

    return model