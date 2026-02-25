from pathlib import Path
import pandas as pd

# -----------------------------
# Data
# -----------------------------
from src.data_loader import (
    detect_available_seasons,
    build_multi_season_dataset,
    build_game_level_spread_dataset
)

# -----------------------------
# Points Model (Generative)
# -----------------------------
from src.models.points.model_wrapper import build_model_from_df
from src.models.points.diagnostics import run_diagnostics
from src.fit_model import sample_model
from src.evaluation.backtests.rolling_backtest_points import rolling_backtest_points

# -----------------------------
# Spread Model (Direct)
# -----------------------------
from src.models.spread.spread_model_wrapper import build_spread_model_from_df
from src.evaluation.backtests.rolling_backtest_spread import rolling_backtest_spread

# -----------------------------
# Reporting
# -----------------------------
from src.evaluation.reporting import (
    summarize_points_results,
    summarize_spread_results,
    print_summary
)

# -----------------------------
# Benchmarks
# -----------------------------
from src.evaluation.vegas_loader import load_vegas_spreads
from src.evaluation.benchmarks.elo_model import run_elo_backtest

#------------------------------
# Metric Calculation
#------------------------------
from src.evaluation.metrics import rmse, mae, brier_score

# -----------------------------
# Logging
# -----------------------------
from src.experiment_logger import log_experiment

# =====================================================
# PIPELINE CONFIGURATION
# =====================================================
CONFIG = {
    "force_rebuild": False,
    "run_points": False,
    "run_spread": True,
    "multi_season": True,
    "min_year": 2018,
    "dataset_version": "v1"
}

# =====================================================
# SPREAD MODEL EXPERIMENTS (A: Beta(8,2) vs Beta(2,2))
# =====================================================
SPREAD_EXPERIMENTS = [
    {
        "name": "Beta(8,2)",
        "rho_prior": {"type": "beta", "a": 8, "b": 2},
        "sigma_theta_prior": {"type": "halfnormal", "scale": 0.2}
    },
    {
        "name": "Beta(2,2)",
        "rho_prior": {"type": "beta", "a": 2, "b": 2},
        "sigma_theta_prior": {"type": "halfnormal", "scale": 0.2}
    }
]

# =====================================================
# MAIN PIPELINE
# =====================================================
def main(config):

    force_rebuild = config["force_rebuild"]
    run_points = config["run_points"]
    run_spread = config["run_spread"]

    # -------------------------------------------------
    # Dataset construction
    # -------------------------------------------------
    YEARS = detect_available_seasons(min_year=config["min_year"])
    DATA_VERSION = config["dataset_version"]

    data_path = Path(f"data/processed/nfl_dynamic_ready_{DATA_VERSION}.csv")

    if force_rebuild or not data_path.exists():
        print("Building dataset...")
        data_path = build_multi_season_dataset(
            YEARS,
            version=DATA_VERSION,
            force_rebuild=False
        )

    df = pd.read_csv(data_path)

    # -------------------------------------------------
    # Merge Vegas spreads
    # -------------------------------------------------
    vegas = load_vegas_spreads()
    df = df.merge(vegas, on="game_id", how="left")

    print("Vegas missing %:",
          df["closing_spread"].isna().mean())

    YEARS = sorted(df["season"].unique())
    TEST_SEASON = YEARS[-1]

    print(f"Train seasons: {YEARS[:-1]}")
    print(f"Test season: {TEST_SEASON}")

    all_results = {}

    # =====================================================
    # POINTS MODEL
    # =====================================================
    if run_points:

        print("\nRunning POINTS model...")

        # Full structural fit
        full_model = build_model_from_df(df)
        full_trace = sample_model(full_model)

        team_to_idx = (
            df[["team", "team_idx"]]
            .drop_duplicates()
            .set_index("team")["team_idx"]
            .to_dict()
        )

        diag = run_diagnostics(full_trace, team_to_idx)

        print("\nPOINTS MODEL STRUCTURE")
        print("-----------------------")

        for key, val in diag.items():
            if isinstance(val, float):
                print(f"{key}: {val:.4f}")

        # Rolling backtest
        points_results = rolling_backtest_points(
            df,
            build_model_from_df,
            sample_model,
            TEST_SEASON
        )

        points_summary = summarize_points_results(points_results)
        print_summary(points_summary, "POINTS MODEL SUMMARY")

        all_results["points_model"] = points_summary

    # =====================================================
    # SPREAD MODEL EXPERIMENT LOOP
    # =====================================================
    if run_spread:

        print("\nRunning SPREAD model experiments...")
        spread_df = build_game_level_spread_dataset(df)

        spread_experiment_results = {}

        for experiment in SPREAD_EXPERIMENTS:

            print("\n----------------------------------------")
            print(f"Experiment: {experiment['name']}")
            print("----------------------------------------")

            spread_results = rolling_backtest_spread(
                spread_df,
                build_spread_model_from_df,
                sample_model,
                TEST_SEASON,
                experiment
            )

            spread_summary = summarize_spread_results(spread_results)

            print_summary(
                spread_summary,
                f"SPREAD MODEL — {experiment['name']}"
            )

            spread_experiment_results[experiment["name"]] = spread_summary

        all_results["spread_model"] = spread_experiment_results

        # Optional quick comparison
        print("\nSPREAD EXPERIMENT COMPARISON (RMSE)")
        print("------------------------------------")
        for name, summary in spread_experiment_results.items():
            print(f"{name}: {summary['rmse_mean']:.4f}")

    # =====================================================
    # ELO BENCHMARK
    # =====================================================
    elo_results = run_elo_backtest(df, TEST_SEASON)
    elo_summary = {
        "rmse": rmse(
            elo_results["predicted_spread"].values,
            elo_results["observed_spread"].values
        ),
        "mae": mae(
            elo_results["predicted_spread"].values,
            elo_results["observed_spread"].values
        ),
        "brier": brier_score(
            elo_results["prob_home_win"].values,
            elo_results["observed_spread"].values
        )
    }

    print_summary(elo_summary, "ELO BENCHMARK")

    all_results["elo"] = elo_summary

    # =====================================================
    # LOG EXPERIMENT
    # =====================================================
    run_config = {
        "years": YEARS,
        "dataset_version": DATA_VERSION,
        "evaluation": "rolling_expanding_window"
    }

    log_experiment(run_config, all_results, {})

    return all_results

if __name__ == "__main__":
    main(CONFIG)