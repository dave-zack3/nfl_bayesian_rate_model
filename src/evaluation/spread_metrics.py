import numpy as np
import pandas as pd
from scipy.stats import norm

def spread_rmse(pred_mean, observed):
    return np.sqrt(np.mean((pred_mean - observed) ** 2))

def spread_mae(pred_mean, observed):
    return np.mean(np.abs(pred_mean - observed))

def brier_score(prob_home_win, observed_spread):
    observed_win = (observed_spread > 0).astype(int)
    return np.mean((prob_home_win - observed_win) ** 2)

def log_predictive_density(spread_samples, observed_spread):
    """
    Approximate log predictive density via Monte Carlo.
    """
    log_probs = []
    for i in range(len(observed_spread)):
        samples = spread_samples[:, i]
        kde_std = np.std(samples)
        kde_mean = np.mean(samples)
        log_probs.append(norm.logpdf(
            observed_spread[i],
            loc=kde_mean,
            scale=kde_std + 1e-6
        ))
    return np.mean(log_probs)

def betting_edge(prob_cover, observed_spread, vegas_line):

    actual_cover = (observed_spread > vegas_line).astype(int)

    implied_prob = 0.5  # assume -110 both sides

    edge = prob_cover - implied_prob

    profit = []
    for p, actual in zip(prob_cover, actual_cover):

        if p > 0.52:  # simple threshold
            if actual == 1:
                profit.append(0.91)
            else:
                profit.append(-1)
        else:
            profit.append(0)

    return sum(profit), np.mean(edge)

def calibration_curve(prob_home_win, observed_spread, bins=10):

    observed_win = (observed_spread > 0).astype(int)

    df = pd.DataFrame({
        "prob": prob_home_win,
        "actual": observed_win
    })

    df["bin"] = pd.qcut(df["prob"], bins, duplicates="drop")

    calibration = df.groupby("bin").agg(
        mean_prob=("prob", "mean"),
        actual_rate=("actual", "mean")
    )

    return calibration

def sharpness(spread_samples):
    return spread_samples.var()