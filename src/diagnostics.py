import arviz as az
import numpy as np
import pandas as pd

def run_diagnostics(trace, team_to_idx):

    posterior = trace.posterior

    off_raw = posterior["offense_raw"].values
    def_raw = posterior["defense_raw"].values

    sigma_off = posterior["sigma_off"].values
    sigma_def = posterior["sigma_def"].values

    offense = off_raw * sigma_off[..., None]
    defense = def_raw * sigma_def[..., None]

    offense = offense - offense.mean(axis=-1, keepdims=True)
    defense = defense - defense.mean(axis=-1, keepdims=True)

    off_mean = offense.mean(axis=(0,1))
    def_mean = defense.mean(axis=(0,1))

    idx_to_team = {v: k for k, v in team_to_idx.items()}

    team_effects = pd.DataFrame({
        "team_idx": range(len(off_mean)),
        "offense_effect": off_mean,
        "defense_effect": def_mean,
        "offense_rate_ratio": np.exp(off_mean),
        "defense_rate_ratio": np.exp(-def_mean)
    })

    team_effects["team"] = team_effects["team_idx"].map(idx_to_team)
    team_effects["offense_rank"] = team_effects["offense_rate_ratio"].rank(ascending=False)
    team_effects["defense_rank"] = team_effects["defense_rate_ratio"].rank(ascending=False)

    team_effects.sort_values("offense_effect", ascending=False)
    team_effects.sort_values("defense_effect")

    summary = az.summary(trace)
    return {
        "summary": summary,
        "team_effects": team_effects
    }