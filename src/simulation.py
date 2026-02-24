import numpy as np
import pymc as pm

def run_ppc(model, trace, df):

    with model:
        ppc = pm.sample_posterior_predictive(
            trace,
            var_names=["points"],
            random_seed=3
        )

    y_rep = (
        ppc.posterior_predictive["points"]
        .stack(sample=("chain", "draw"))
        .values
    )  # (samples, n_observations)

    # --------------------------------------------------
    # Spread Computation
    # --------------------------------------------------

    # Identify home rows
    home_mask = df["home_flag"].values == 1
    away_mask = df["home_flag"].values == 0

    home_points = y_rep[:, home_mask]
    away_points = y_rep[:, away_mask]

    spread_samples = home_points - away_points

    # Observed spreads
    observed_home = df.loc[home_mask, "points"].values
    observed_away = df.loc[away_mask, "points"].values
    observed_spread = observed_home - observed_away

    # Posterior mean spread
    mean_pred_spread = spread_samples.mean(axis=0)

    # Spread variance
    spread_variance = spread_samples.flatten().var()

    # 94% CI width
    lower_spread = np.percentile(spread_samples, 3, axis=0)
    upper_spread = np.percentile(spread_samples, 97, axis=0)
    avg_spread_ci_width = (upper_spread - lower_spread).mean()

    # Probability home team wins
    prob_home_win = (spread_samples > 0).mean(axis=0).mean()

    observed = df["points"].values

    # --- Mean ---
    observed_mean = observed.mean()
    simulated_mean = y_rep.mean()

    # --- Variance (flattened for apples-to-apples) ---
    observed_var = observed.var()
    simulated_var = y_rep.flatten().var()

    # --- 94% Predictive CI Width ---
    lower = np.percentile(y_rep, 3, axis=0)
    upper = np.percentile(y_rep, 97, axis=0)
    avg_ci_width = (upper - lower).mean()

    # --- Tail frequencies ---
    observed_tail_40 = np.mean(observed >= 40)
    simulated_tail_40 = np.mean(y_rep >= 40)

    observed_tail_50 = np.mean(observed >= 50)
    simulated_tail_50 = np.mean(y_rep >= 50)

    return {
        "team_game_mean_observed": observed_mean,
        "team_game_mean_simulated": simulated_mean,
        "team_game_var_observed": observed_var,
        "team_game_var_simulated": simulated_var,
        "avg_94pct_predictive_CI_width": avg_ci_width,
        "team_game_tail_40_observed": observed_tail_40,
        "team_game_tail_40_simulated": simulated_tail_40,
        "team_game_tail_50_observed": observed_tail_50,
        "team_game_tail_50_simulated": simulated_tail_50,
        "spread_variance": spread_variance,
        "avg_spread_94pct_CI_width": avg_spread_ci_width,
        "prob_home_win_avg": prob_home_win,
    }