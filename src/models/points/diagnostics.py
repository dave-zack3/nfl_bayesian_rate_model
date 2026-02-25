import arviz as az
import numpy as np
import pandas as pd

def run_diagnostics(trace, team_to_idx):

    posterior = trace.posterior

    results = {}

    # =====================================================
    # Structural parameter summaries
    # (works for both RW and AR(1) models)
    # =====================================================

    if "beta_pace" in posterior:
        results["posterior_mean_beta_pace"] = posterior["beta_pace"].mean().item()
        results["posterior_sd_beta_pace"] = posterior["beta_pace"].std().item()

    if "phi" in posterior:
        results["posterior_mean_phi"] = posterior["phi"].mean().item()
        results["posterior_sd_phi"] = posterior["phi"].std().item()

    # --- Random Walk parameters (old model) ---
    if "sigma_off_rw" in posterior:
        results["posterior_mean_sigma_off_rw"] = posterior["sigma_off_rw"].mean().item()
        results["posterior_sd_sigma_off_rw"] = posterior["sigma_off_rw"].std().item()

    if "sigma_def_rw" in posterior:
        results["posterior_mean_sigma_def_rw"] = posterior["sigma_def_rw"].mean().item()
        results["posterior_sd_sigma_def_rw"] = posterior["sigma_def_rw"].std().item()

    # --- AR(1) parameters (new model) ---
    if "rho" in posterior:
        results["posterior_mean_rho"] = posterior["rho"].mean().item()
        results["posterior_sd_rho"] = posterior["rho"].std().item()

    if "sigma_off" in posterior:
        results["posterior_mean_sigma_off"] = posterior["sigma_off"].mean().item()
        results["posterior_sd_sigma_off"] = posterior["sigma_off"].std().item()

    if "sigma_def" in posterior:
        results["posterior_mean_sigma_def"] = posterior["sigma_def"].mean().item()
        results["posterior_sd_sigma_def"] = posterior["sigma_def"].std().item()

    # =====================================================
    # Dynamic team effects reconstruction
    # =====================================================

    # Support both naming conventions
    if "offense_raw" in posterior:
        offense = posterior["offense_raw"].values
    else:
        offense = posterior["offense_raw"].values

    if "defense_raw" in posterior:
        defense = posterior["defense_raw"].values
    else:
        defense = posterior["defense_raw"].values

    # Enforce sum-to-zero per week
    offense = offense - offense.mean(axis=2, keepdims=True)
    defense = defense - defense.mean(axis=2, keepdims=True)

    # Average over chains + draws
    off_mean = offense.mean(axis=(0,1))  # (team, week)
    def_mean = defense.mean(axis=(0,1))  # (team, week)

    n_teams, n_weeks = off_mean.shape

    idx_to_team = {v: k for k, v in team_to_idx.items()}

    rows = []
    for team_idx in range(n_teams):
        for week in range(n_weeks):
            rows.append({
                "team_idx": team_idx,
                "team": idx_to_team[team_idx],
                "week": week,
                "offense_effect": off_mean[team_idx, week],
                "defense_effect": def_mean[team_idx, week],
                "offense_rate_ratio": np.exp(off_mean[team_idx, week]),
                "defense_rate_ratio": np.exp(-def_mean[team_idx, week]),
            })

    team_effects = pd.DataFrame(rows)

    results["team_effects"] = team_effects

    # =====================================================
    # Dynamic offense CI width
    # =====================================================

    off_samples = (
        posterior["offense_raw"]
        .stack(sample=("chain", "draw"))
        .values
    )

    # Ensure samples are last axis
    if off_samples.shape[0] != n_teams:
        off_samples = np.moveaxis(off_samples, 0, -1)

    lower = np.percentile(off_samples, 3, axis=-1)
    upper = np.percentile(off_samples, 97, axis=-1)

    results["avg_offense_ci_width"] = (upper - lower).mean()

    # =====================================================
    # AR(1) identifiability diagnostic (optional but powerful)
    # =====================================================

    if "rho" in posterior and "sigma_off" in posterior:
        rho_samples = posterior["rho"].stack(sample=("chain", "draw")).values
        sigma_samples = posterior["sigma_off"].stack(sample=("chain", "draw")).values
        corr = np.corrcoef(rho_samples, sigma_samples)[0,1]
        results["rho_sigma_corr"] = corr

    if "alpha" in posterior:
        print(f"alpha (mean ± sd): "
            f"{posterior['alpha'].mean().item():.4f} ± "
            f"{posterior['alpha'].std().item():.4f}")

    # =====================================================
    # ArviZ summary table
    # =====================================================

    results["summary"] = az.summary(trace)

    return results