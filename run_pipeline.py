from src.model_spec import build_model
from src.fit_model import sample_model
from src.diagnostics import run_diagnostics
import pandas as pd
import numpy as np

def main():

    df = pd.read_csv("data/processed/nfl_2021_team_game.csv")

    team_idx = df["team_idx"].values
    opp_idx = df["opponent_idx"].values
    points = df["points"].values
    drives = df["drives"].values
    home_flag = df["home_flag"].values

    n_teams = df["team_idx"].nunique()

    model = build_model(
        team_idx,
        opp_idx,
        points,
        drives,
        home_flag,
        n_teams
    )

    trace = sample_model(model)

    summary = run_diagnostics(trace)
    print(summary)

if __name__ == "__main__":
    main()