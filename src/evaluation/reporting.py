def summarize_points_results(results_df):
    return {
        "rmse_mean": results_df["rmse"].mean(),
        "mae_mean": results_df["mae"].mean(),
        "brier_mean": results_df["brier"].mean(),
        "lpd_mean": results_df["lpd"].mean(),
        "units_total": results_df["units_won"].sum(),
        "avg_edge_mean": results_df["avg_edge"].mean(),
        "sharpness_mean": results_df["sharpness"].mean(),
    }


def summarize_spread_results(results_df):
    return {
        "rmse_mean": results_df["rmse"].mean(),
        "mae_mean": results_df["mae"].mean(),
        "lpd_mean": results_df["lpd"].mean(),
        "sharpness_mean": results_df["sharpness"].mean(),
    }


def print_summary(summary_dict, title):
    print("\n" + "=" * 40)
    print(title)
    print("=" * 40)

    for k, v in summary_dict.items():
        print(f"{k}: {v:.4f}")