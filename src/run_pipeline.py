from pathlib import Path
import pandas as pd
import arviz as az

# -----------------------------
# Data
# -----------------------------
from src.data_loader import (
    detect_available_seasons,
    build_multi_season_dataset,
    build_game_level_spread_dataset
)

# -----------------------------
# Points Model
# -----------------------------
from src.models.points.model_wrapper import build_model_from_df
from src.models.points.diagnostics import run_diagnostics
from src.fit_model import sample_model
from src.evaluation.backtests.rolling_backtest_points import rolling_backtest_points

# -----------------------------
# Spread Model
# -----------------------------
from src.models.spread.spread_model_wrapper import build_spread_model_from_df
from src.models.spread.diagnostics import summarize_spread_structure
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

# -----------------------------
# Metrics
# -----------------------------
from src.evaluation.metrics import rmse, mae, brier_score

# -----------------------------
# Logging
# -----------------------------
from src.experiment_logger import log_experiment


# =====================================================
# CONFIGURATION
# =====================================================
CONFIG = {
    "force_rebuild": False,
    "run_points": False,
    "run_spread": True,
    "min_year": 2018,
    "dataset_version": "v1",
    "draws": 500,
    "tune": 500,
    "chains": 2
}

# =====================================================
# SPREAD EXPERIMENTS
# =====================================================
SPREAD_EXPERIMENTS = [
    {
        "name": "Bayesian_Heteroskedastic",
        "rho_prior": {"type": "beta", "a": 8, "b": 2},
        "sigma_theta_prior": {"type": "halfnormal", "scale": 0.2},
        "noise_type": "heteroskedastic"
    },
    {
        "name": "Bayesian_Homoskedastic",
        "rho_prior": {"type": "beta", "a": 8, "b": 2},
        "sigma_theta_prior": {"type": "halfnormal", "scale": 0.2},
        "noise_type": "homoskedastic"
    }
]


# =====================================================
# MAIN
# =====================================================
def main(config):

    force_rebuild = config["force_rebuild"]
    run_points = config["run_points"]
    run_spread = config["run_spread"]

    # -------------------------------------------------
    # Dataset
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

    vegas = load_vegas_spreads()
    df = df.merge(vegas, on="game_id", how="left")

    YEARS = sorted(df["season"].unique())
    TEST_SEASON = YEARS[-1]

    print(f"\nTrain seasons: {YEARS[:-1]}")
    print(f"Test season: {TEST_SEASON}")

    all_results = {}

    # =====================================================
    # SPREAD MODEL
    # =====================================================
    if run_spread:

        print("\nRunning SPREAD model experiments...")
        spread_df = build_game_level_spread_dataset(df)

        spread_experiment_results = {}

        for experiment in SPREAD_EXPERIMENTS:

            print("\n=================================================")
            print(f"STRUCTURAL FIT — {experiment['name']}")
            print("=================================================")

            # -------------------------------------------------
            # FULL STRUCTURAL FIT (ALL DATA)
            # -------------------------------------------------
            full_model = build_spread_model_from_df(spread_df, experiment)
            full_trace = sample_model(
                full_model,
                draws=config["draws"],
                tune=config["tune"],
                chains=config["chains"]
            )

            # Create output folder once
            output_dir = Path("whitepaper_outputs")
            output_dir.mkdir(exist_ok=True)

            # Save trace
            az.to_netcdf(
                full_trace,
                output_dir / f"{experiment['name']}_trace.nc"
            )

            print(f"Trace saved: {experiment['name']}_trace.nc")

            structure_summary = summarize_spread_structure(full_trace)

            for param, stats in structure_summary.items():
                if isinstance(stats, dict) and "mean" in stats:
                    if "sd" in stats:
                        print(f"{param}: {stats['mean']:.4f} ± {stats['sd']:.4f}")
                    else:
                        print(f"{param}: {stats['mean']:.4f}")

            # -------------------------------------------------
            # ROLLING BACKTEST
            # -------------------------------------------------
            print("\nRolling Backtest...")

            spread_results = rolling_backtest_spread(
                spread_df,
                build_spread_model_from_df,
                sample_model,
                TEST_SEASON,
                experiment
            )

            spread_results.to_csv(
                output_dir / f"{experiment['name']}_rolling.csv",
                index=False
            )

            performance_summary = summarize_spread_results(spread_results)

            print_summary(
                performance_summary,
                f"SPREAD PERFORMANCE — {experiment['name']}"
            )

            spread_experiment_results[experiment["name"]] = {
                "structure": structure_summary,
                "performance": performance_summary
            }

        all_results["spread_model"] = spread_experiment_results

        # Quick RMSE comparison
        print("\nRMSE Comparison")
        print("----------------")
        for name, res in spread_experiment_results.items():
            print(f"{name}: {res['performance']['rmse_mean']:.4f}")

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

    # -------------------------------------------------
    # Log
    # -------------------------------------------------
    run_config = {
        "years": YEARS,
        "dataset_version": DATA_VERSION,
        "evaluation": "rolling_expanding_window",
        "draws": config["draws"],
        "tune": config["tune"]
    }

    log_experiment(run_config, all_results, {})

    return all_results

if __name__ == "__main__":
    main(CONFIG)