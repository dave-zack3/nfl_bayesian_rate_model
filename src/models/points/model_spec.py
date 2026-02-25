import numpy as np
import pymc as pm

def build_model(team_idx, opp_idx, time_idx,
                points, drives, home_flag,
                n_teams, n_weeks):

    with pm.Model() as model:

        # -------------------------------------------------
        # AR(1) persistence parameter (global)
        # -------------------------------------------------
        # Constrained to (0,1) since negative persistence
        # is not meaningful for team strength.
        rho = pm.Beta("rho", alpha=2, beta=2)

        # -------------------------------------------------
        # Innovation scales
        # -------------------------------------------------
        sigma_off = pm.HalfNormal("sigma_off", 0.3)
        sigma_def = pm.HalfNormal("sigma_def", 0.3)

        # -------------------------------------------------
        # Initial week latent strengths
        # -------------------------------------------------
        offense_init = pm.Normal(
            "offense_init",
            mu=0,
            sigma=0.3,
            shape=n_teams
        )

        defense_init = pm.Normal(
            "defense_init",
            mu=0,
            sigma=0.3,
            shape=n_teams
        )

        # -------------------------------------------------
        # Innovation noise across weeks
        # -------------------------------------------------
        offense_eps = pm.Normal(
            "offense_eps",
            mu=0,
            sigma=sigma_off,
            shape=(n_teams, n_weeks - 1)
        )

        defense_eps = pm.Normal(
            "defense_eps",
            mu=0,
            sigma=sigma_def,
            shape=(n_teams, n_weeks - 1)
        )

        # -------------------------------------------------
        # Build AR(1) recursively
        # -------------------------------------------------
        offense_states = [offense_init]
        defense_states = [defense_init]

        for t in range(1, n_weeks):
            offense_t = rho * offense_states[t-1] + offense_eps[:, t-1]
            defense_t = rho * defense_states[t-1] + defense_eps[:, t-1]

            offense_states.append(offense_t)
            defense_states.append(defense_t)

        offense_raw = pm.Deterministic(
            "offense_raw",
            pm.math.stack(offense_states, axis=1)
        )

        defense_raw = pm.Deterministic(
            "defense_raw",
            pm.math.stack(defense_states, axis=1)
        )

        # -------------------------------------------------
        # Sum-to-zero constraint per week
        # -------------------------------------------------
        offense = offense_raw - pm.math.mean(
            offense_raw, axis=0, keepdims=True
        )

        defense = defense_raw - pm.math.mean(
            defense_raw, axis=0, keepdims=True
        )

        # -------------------------------------------------
        # Other structural parameters
        # -------------------------------------------------
        intercept = pm.Normal("intercept", 0, 1)

        sigma_home = pm.HalfNormal("sigma_home", 0.5)
        home_raw = pm.Normal("home_raw", 0, 1, shape=n_teams)
        home_effect = home_raw * sigma_home

        beta_pace = pm.Normal("beta_pace", 1.0, 0.3)

        phi = pm.HalfNormal("phi", 5)

        # -------------------------------------------------
        # Linear predictor
        # -------------------------------------------------
        log_lambda = (
            intercept
            + offense[team_idx, time_idx]
            - defense[opp_idx, time_idx]
            + home_effect[team_idx] * home_flag
            + beta_pace * np.log(drives)
        )

        pm.NegativeBinomial(
            "points",
            mu=pm.math.exp(log_lambda),
            alpha=phi,
            observed=points
        )

    return model