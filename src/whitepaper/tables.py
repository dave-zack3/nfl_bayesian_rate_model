import pandas as pd

def structural_table(all_results):
    rows = []

    for name, res in all_results["spread_model"].items():
        struct = res["structure"]

        rows.append({
            "Model": name,
            "rho": struct["rho"]["mean"],
            "sigma_weekly": struct["sigma_weekly"]["mean"],
            "sigma_offseason": struct["sigma_offseason"]["mean"],
            "beta_home": struct["beta_home"]["mean"],
            "avg_abs_strength": struct["avg_abs_strength"]["mean"]
        })

    return pd.DataFrame(rows)

def performance_table(all_results):
    rows = []

    for name, res in all_results["spread_model"].items():
        perf = res["performance"]

        rows.append({
            "Model": name,
            "RMSE": perf["rmse_mean"],
            "MAE": perf["mae_mean"],
            "LPD": perf["lpd_mean"],
            "Sharpness": perf["sharpness_mean"]
        })

    rows.append({
        "Model": "Elo",
        "RMSE": all_results["elo"]["rmse"],
        "MAE": all_results["elo"]["mae"],
        "LPD": None,
        "Sharpness": None
    })

    return pd.DataFrame(rows)