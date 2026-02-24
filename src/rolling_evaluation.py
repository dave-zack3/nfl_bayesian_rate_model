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

        train_df = df[
            (df["season"] < season) |
            ((df["season"] == season) & (df["time_idx"] < week))
        ]

        test_df = season_df[season_df["time_idx"] == week]

        model = build_model_fn(train_df)
        trace = sample_fn(model)

        # Posterior predictive on test_df
        # Compute log predictive density
        # Compute MSE
        # Compute CRPS (optional)

        results.append({
            "week": week,
            "n_test_games": len(test_df),
            # Add metrics here
        })

    return pd.DataFrame(results)