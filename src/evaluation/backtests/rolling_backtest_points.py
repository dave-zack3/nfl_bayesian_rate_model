import pandas as pd
import numpy as np
import pymc as pm
from src.evaluation.metrics import (
    mae,
    brier_score,
    log_predictive_density,
    betting_edge
)

def rolling_backtest_points(
    df,
    build_model_fn,
    sample_fn,
    target_season
):

    season_df = df[df["season"] == target_season]
    weeks = sorted(season_df["time_idx"].unique())

    results = []

    for week in weeks[1:]:

        print(f"\nRolling forecast: Season {target_season}, Week {week}")

        # ----------------------------------------
        # Split train vs current test week
        # ----------------------------------------
        train_df = df[
            (df["season"] < target_season) |
            ((df["season"] == target_season) &
             (df["time_idx"] < week))
        ]

        test_df = season_df[
            season_df["time_idx"] == week
        ]

        # ----------------------------------------
        # Combine train + test
        # Keep state dimension fixed
        # ----------------------------------------
        combined_df = pd.concat([train_df, test_df]).copy()

        # Mask test observations
        combined_df.loc[
            combined_df.index.isin(test_df.index),
            "points"
        ] = np.nan

        # ----------------------------------------
        # Fit model on combined data
        # ----------------------------------------
        model = build_model_fn(combined_df)
        trace = sample_fn(model)

        # ----------------------------------------
        # Posterior predictive for all rows
        # Missing rows automatically predicted
        # ----------------------------------------
        with model:
            ppc = pm.sample_posterior_predictive(
                trace,
                var_names=["points"]
            )

        pp_array = ppc.posterior_predictive["points"]

        # Stack chain/draw
        pp_stacked = pp_array.stack(sample=("chain", "draw"))

        # Ensure dimension order
        pp_stacked = pp_stacked.transpose("sample", "points_dim_0")

        y_rep = pp_stacked.values

        # ----------------------------------------
        # Identify which rows correspond to test set
        # ----------------------------------------
        test_mask = combined_df["points"].isna().values

        y_test_rep = y_rep[:, test_mask]

        test_rows = combined_df.loc[test_mask]

        home_mask = test_rows["home_flag"].values == 1
        away_mask = test_rows["home_flag"].values == 0

        home_points = y_test_rep[:, home_mask]
        away_points = y_test_rep[:, away_mask]

        spread_samples = home_points - away_points

        observed_home = test_df.loc[
            test_df["home_flag"] == 1,
            "points"
        ].values

        observed_away = test_df.loc[
            test_df["home_flag"] == 0,
            "points"
        ].values

        observed_spread = observed_home - observed_away

        mean_spread = spread_samples.mean(axis=0)
        prob_home_win = (spread_samples > 0).mean(axis=0)

        vegas_line = test_df.loc[
            test_df["home_flag"] == 1,
            "closing_spread"
        ].values

        prob_cover = (spread_samples > vegas_line).mean(axis=0)

        profit, avg_edge = betting_edge(
            prob_cover,
            observed_spread,
            vegas_line
        )

        metrics = {
            "week": week,
            "rmse": rmse(mean_spread, observed_spread),
            "mae": mae(mean_spread, observed_spread),
            "brier": brier_score(prob_home_win, observed_spread),
            "lpd": log_predictive_density(
                spread_samples,
                observed_spread
            ),
            "units_won": profit,
            "avg_edge": avg_edge,
            "sharpness": spread_samples.var()
        }

        results.append(metrics)

    return pd.DataFrame(results)