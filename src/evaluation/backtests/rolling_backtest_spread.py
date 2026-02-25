import pandas as pd
import pymc as pm
import numpy as np
from src.evaluation.metrics import (
    rmse,
    mae,
    log_predictive_density
)

def rolling_backtest_spread(
    df,
    build_model_fn,
    sample_fn,
    target_season,
    model_config
):
    """
    Rolling expanding-window backtest for direct spread model.

    Parameters
    ----------
    df : DataFrame
        Game-level spread dataset
    build_model_fn : function
        Wrapper that builds PyMC model from df + config
    sample_fn : function
        Sampling function (NUTS)
    target_season : int
        Season to evaluate out-of-sample
    model_config : dict
        Prior + noise configuration
    """

    season_df = df[df["season"] == target_season]
    weeks = sorted(season_df["time_idx"].unique())

    results = []

    for week in weeks[1:]:

        print(f"\nSpread model forecast: Season {target_season}, Week {week}")
        print(f"Experiment: {model_config.get('name', 'unnamed')}")

        # =====================================================
        # Train/Test Split (NO LEAKAGE)
        # =====================================================

        train_df = df[
            (df["season"] < target_season) |
            (
                (df["season"] == target_season) &
                (df["time_idx"] < week)
            )
        ]

        test_df = season_df[
            season_df["time_idx"] == week
        ]

        # ----------------------------------------
        # Combine train + test
        # ----------------------------------------

        combined_df = pd.concat([train_df, test_df]).copy()

        # Mask test observations
        combined_df.loc[test_df.index, "spread"] = np.nan

        # ----------------------------------------
        # Build + fit on combined data
        # ----------------------------------------

        model = build_model_fn(combined_df, model_config)
        trace = sample_fn(model)

        # ----------------------------------------
        # Posterior predictive
        # ----------------------------------------

        with model:
            ppc = pm.sample_posterior_predictive(
                trace,
                var_names=["spread_obs"],
                random_seed=42
            )

        y_rep = (
            ppc.posterior_predictive["spread_obs"]
            .stack(sample=("chain", "draw"))
            .transpose("sample", "spread_obs_dim_0")
            .values
        )

        # Identify test rows
        test_mask = combined_df["spread"].isna().values

        y_test_rep = y_rep[:, test_mask]
        observed = test_df["spread"].values

        mean_pred = y_test_rep.mean(axis=0)

        # =====================================================
        # Metrics
        # =====================================================

        metrics = {
            "week": week,
            "rmse": rmse(mean_pred, observed),
            "mae": mae(mean_pred, observed),
            "lpd": log_predictive_density(y_test_rep, observed),
            "sharpness": y_test_rep.var()
        }

        results.append(metrics)

    return pd.DataFrame(results)