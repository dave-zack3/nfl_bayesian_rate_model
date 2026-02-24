import numpy as np

def compute_elo_metrics(df):

    rmse = np.sqrt(np.mean(
        (df["predicted_spread"] - df["observed_spread"]) ** 2
    ))

    mae = np.mean(
        np.abs(df["predicted_spread"] - df["observed_spread"])
    )

    observed_win = (df["observed_spread"] > 0).astype(int)

    brier = np.mean(
        (df["prob_home_win"] - observed_win) ** 2
    )

    return {
        "rmse": rmse,
        "mae": mae,
        "brier": brier
    }