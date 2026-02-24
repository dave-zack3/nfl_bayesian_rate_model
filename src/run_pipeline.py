from pathlib import Path
import pandas as pd
from src.data_loader import detect_available_seasons, build_multi_season_dataset
from src.model_wrapper import build_model_from_df
from src.fit_model import sample_model
from src.evaluation.rolling_backtest import rolling_backtest
from src.evaluation.vegas_loader import load_vegas_spreads
from src.experiment_logger import log_experiment
from src.evaluation.elo_model import run_elo_backtest
from src.evaluation.elo_metrics import compute_elo_metrics

def main(force_rebuild=False):

    YEARS = detect_available_seasons(min_year=2021)
    DATA_VERSION = "v1"

    data_path = Path(f"data/processed/nfl_dynamic_ready_{DATA_VERSION}.csv")

    if force_rebuild or not data_path.exists():
        print("Building dataset...")
        data_path = build_multi_season_dataset(
            YEARS,
            version=DATA_VERSION,
            force_rebuild=False
        )

    df = pd.read_csv(data_path)

    # -----------------------------
    # Merge Vegas
    # -----------------------------

    vegas = load_vegas_spreads()
    df = df.merge(vegas, on="game_id", how="left")

    print("Vegas missing %:",
      df["closing_spread"].isna().mean())

    # -----------------------------
    # Train/Test split
    # -----------------------------

    YEARS = sorted(df["season"].unique())
    TRAIN_SEASONS = YEARS[:-1]
    TEST_SEASON = YEARS[-1]

    print(f"Train seasons: {TRAIN_SEASONS}")
    print(f"Test season: {TEST_SEASON}")

    # -----------------------------
    # Rolling Backtest
    # -----------------------------

    results_df = rolling_backtest(
        df,
        build_model_from_df,
        sample_model,
        TEST_SEASON
    )

    summary = {
        "rmse_mean": results_df["rmse"].mean(),
        "mae_mean": results_df["mae"].mean(),
        "brier_mean": results_df["brier"].mean(),
        "lpd_mean": results_df["lpd"].mean(),
        "units_total": results_df["units_won"].sum(),
        "avg_edge_mean": results_df["avg_edge"].mean(),
        "sharpness_mean": results_df["sharpness"].mean(),
    }

    print("\n=== ROLLING BACKTEST SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")

    # -----------------------------
    # Log experiment
    # -----------------------------

    config = {
        "years": YEARS,
        "dataset_version": DATA_VERSION,
        "model_type": "dynamic_nb",
        "evaluation": "rolling_expanding_window"
    }

    log_experiment(config, summary, results_df.to_dict())

    # -----------------------------
    # Elo Benchmark
    # -----------------------------

    elo_results = run_elo_backtest(df, TEST_SEASON)
    elo_summary = compute_elo_metrics(elo_results)

    print("\n=== ELO BENCHMARK ===")
    for k, v in elo_summary.items():
        print(f"{k}: {v:.4f}")

    return results_df, summary


if __name__ == "__main__":
    main()