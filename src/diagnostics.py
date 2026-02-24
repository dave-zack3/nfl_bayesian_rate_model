import arviz as az
import numpy as np
import pandas as pd

def run_diagnostics(trace, team_to_idx):

    posterior = trace.posterior

    # =====================================================
    # Structural parameter summaries (dynamic model)
    # =====================================================

    beta_pace_mean = posterior["beta_pace"].mean().item()
    beta_pace_sd = posterior["beta_pace"].std().item()

    phi_mean = posterior["phi"].mean().item()
    phi_sd = posterior["phi"].std().item()

    sigma_off_rw_mean = posterior["sigma_off_rw"].mean().item()
    sigma_off_rw_sd = posterior["sigma_off_rw"].std().item()

    sigma_def_rw_mean = posterior["sigma_def_rw"].mean().item()
    sigma_def_rw_sd = posterior["sigma_def_rw"].std().item()

    # =====================================================
    # Dynamic team effects reconstruction
    # offense_raw shape: (chain, draw, team, week)
    # =====================================================

    offense = posterior["offense_raw"].values
    defense = posterior["defense_raw"].values

    # Enforce sum-to-zero per week
    offense = offense - offense.mean(axis=2, keepdims=True)
    defense = defense - defense.mean(axis=2, keepdims=True)

    # Average over chains + draws
    off_mean = offense.mean(axis=(0,1))  # (team, week)
    def_mean = defense.mean(axis=(0,1))  # (team, week)

    n_teams, n_weeks = off_mean.shape

    idx_to_team = {v: k for k, v in team_to_idx.items()}

    # Build team-week dataframe
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

    # =====================================================
    # Dynamic offense CI width (across posterior draws)
    # =====================================================

    off_samples = (
        posterior["offense_raw"]
        .stack(sample=("chain", "draw"))
        .values  # (team, week, samples)
    )

    # Move samples axis last if necessary
    if off_samples.shape[0] == n_teams:
        pass
    else:
        off_samples = np.moveaxis(off_samples, 0, -1)

    lower = np.percentile(off_samples, 3, axis=-1)
    upper = np.percentile(off_samples, 97, axis=-1)

    avg_offense_ci_width = (upper - lower).mean()

    # =====================================================
    # ArviZ summary
    # =====================================================

    summary = az.summary(trace)

    return {
        "summary": summary,
        "team_effects": team_effects,
        "avg_offense_ci_width": avg_offense_ci_width,

        # Key structural parameters
        "posterior_mean_beta_pace": beta_pace_mean,
        "posterior_sd_beta_pace": beta_pace_sd,

        "posterior_mean_phi": phi_mean,
        "posterior_sd_phi": phi_sd,

        "posterior_mean_sigma_off_rw": sigma_off_rw_mean,
        "posterior_sd_sigma_off_rw": sigma_off_rw_sd,

        "posterior_mean_sigma_def_rw": sigma_def_rw_mean,
        "posterior_sd_sigma_def_rw": sigma_def_rw_sd,
    }