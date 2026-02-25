import numpy as np
import pandas as pd

def rolling_forecast_evaluation(
    build_model_fn,
    sample_fn,
    df,
    season
):

    season_df = df[df["season"] == season]
    weeks = sorted(season_df["time_idx"].unique())

    results = []

    for week in weeks[1:]:

        print(f"Spread forecast: Season {season}, Week {week}")

        # -----------------------------
        # Train/Test split
        # -----------------------------

        train_df = df[
            (df["season"] < season) |
            ((df["season"] == season) & (df["time_idx"] < week))
        ]

        test_df = season_df[season_df["time_idx"] == week]

        # -----------------------------
        # Fit model
        # -----------------------------

        model = build_model_fn(train_df)
        trace = sample_fn(model)

        posterior = trace.posterior

        # -----------------------------
        # Extract posterior samples
        # -----------------------------

        rho = posterior["rho"].stack(sample=("chain", "draw")).values
        sigma_theta = posterior["sigma_theta"].stack(sample=("chain", "draw")).values
        beta_home = posterior["beta_home"].stack(sample=("chain", "draw")).values
        sigma_spread = posterior["sigma_spread"].stack(sample=("chain", "draw")).values

        theta = posterior["theta"].stack(sample=("chain", "draw")).values
        # shape: (n_teams, n_weeks_train, n_samples)

        theta_last = theta[:, -1, :]
        n_teams, n_samples = theta_last.shape

        # -----------------------------
        # One-step-ahead AR(1)
        # -----------------------------

        eps = np.random.normal(
            loc=0.0,
            scale=sigma_theta,
            size=(n_teams, n_samples)
        )

        theta_next = rho * theta_last + eps

        # -----------------------------
        # Predict spreads
        # -----------------------------

        team_idx = test_df["team_idx"].values
        opp_idx = test_df["opponent_idx"].values
        home_flag = test_df["home_flag"].values
        observed_spread = test_df["spread"].values

        n_games = len(test_df)

        spread_samples = np.zeros((n_games, n_samples))

        for g in range(n_games):
            spread_samples[g, :] = (
                theta_next[team_idx[g], :]
                - theta_next[opp_idx[g], :]
                + beta_home * home_flag[g]
            )

        mean_spread = spread_samples.mean(axis=1)

        # -----------------------------
        # Metrics
        # -----------------------------

        rmse = np.sqrt(np.mean((mean_spread - observed_spread) ** 2))

        log_probs = (
            -0.5 * np.log(2 * np.pi * sigma_spread**2)
            -0.5 * ((observed_spread[:, None] - spread_samples) ** 2)
            / (sigma_spread**2)
        )

        lpd = np.mean(
            np.logaddexp.reduce(log_probs, axis=1) - np.log(n_samples)
        )

        sharpness = np.mean(spread_samples.var(axis=1))

        # -----------------------------
        # Append results
        # -----------------------------

        results.append({
            "week": week,
            "n_games": n_games,
            "rmse": rmse,
            "lpd": lpd,
            "sharpness": sharpness
        })

    return pd.DataFrame(results)