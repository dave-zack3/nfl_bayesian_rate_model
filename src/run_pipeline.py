from pathlib import Path
import pandas as pd
from src.data_loader import detect_available_seasons, build_multi_season_dataset
from src.model_wrapper import build_model_from_df
from src.fit_model import sample_model
from src.diagnostics import run_diagnostics
from src.evaluation.rolling_backtest import rolling_backtest
from src.evaluation.vegas_loader import load_vegas_spreads
from src.evaluation.elo_model import run_elo_backtest
from src.evaluation.elo_metrics import compute_elo_metrics
from src.experiment_logger import log_experiment

def main(force_rebuild=False):

    # =====================================================
    # Dataset construction
    # =====================================================

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

    # =====================================================
    # Merge Vegas spreads
    # =====================================================

    vegas = load_vegas_spreads()
    df = df.merge(vegas, on="game_id", how="left")

    print("Vegas missing %:",
          df["closing_spread"].isna().mean())

    # =====================================================
    # Train/Test split
    # =====================================================

    YEARS = sorted(df["season"].unique())
    TRAIN_SEASONS = YEARS[:-1]
    TEST_SEASON = YEARS[-1]

    print(f"Train seasons: {TRAIN_SEASONS}")
    print(f"Test season: {TEST_SEASON}")

    # =====================================================
    # 1️⃣ FULL-DATA STRUCTURAL FIT (for diagnostics)
    # =====================================================

    print("\nFitting full model for structural diagnostics...")

    full_model = build_model_from_df(df)
    full_trace = sample_model(full_model)

    team_to_idx = (
        df[["team", "team_idx"]]
        .drop_duplicates()
        .set_index("team")["team_idx"]
        .to_dict()
    )

    diag = run_diagnostics(full_trace, team_to_idx)

    print("\n==============================")
    print("=== STRUCTURAL PARAMETERS ===")
    print("==============================")

    if "posterior_mean_rho" in diag:
        print(f"rho (mean ± sd): "
              f"{diag['posterior_mean_rho']:.4f} "
              f"± {diag['posterior_sd_rho']:.4f}")

    if "posterior_mean_sigma_off" in diag:
        print(f"sigma_off (mean ± sd): "
              f"{diag['posterior_mean_sigma_off']:.4f} "
              f"± {diag['posterior_sd_sigma_off']:.4f}")

    if "posterior_mean_sigma_def" in diag:
        print(f"sigma_def (mean ± sd): "
              f"{diag['posterior_mean_sigma_def']:.4f} "
              f"± {diag['posterior_sd_sigma_def']:.4f}")

    if "rho_sigma_corr" in diag:
        print(f"Corr(rho, sigma_off): "
              f"{diag['rho_sigma_corr']:.4f}")

    print(f"phi (mean ± sd): "
          f"{diag['posterior_mean_phi']:.4f} "
          f"± {diag['posterior_sd_phi']:.4f}")

    print(f"beta_pace (mean ± sd): "
          f"{diag['posterior_mean_beta_pace']:.4f} "
          f"± {diag['posterior_sd_beta_pace']:.4f}")

    print(f"Avg offense 94% CI width: "
          f"{diag['avg_offense_ci_width']:.4f}")

    # =====================================================
    # 2️⃣ Rolling Backtest (true out-of-sample evaluation)
    # =====================================================

    print("\nRunning rolling backtest...")

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

    # =====================================================
    # 3️⃣ Elo Benchmark
    # =====================================================

    elo_results = run_elo_backtest(df, TEST_SEASON)
    elo_summary = compute_elo_metrics(elo_results)

    print("\n=== ELO BENCHMARK ===")
    for k, v in elo_summary.items():
        print(f"{k}: {v:.4f}")

    # =====================================================
    # 4️⃣ Log Experiment
    # =====================================================

    config = {
        "years": YEARS,
        "dataset_version": DATA_VERSION,
        "model_type": "dynamic_nb_ar1",
        "evaluation": "rolling_expanding_window"
    }

    log_experiment(config, summary, results_df.to_dict())

    return results_df, summary


if __name__ == "__main__":
    main()