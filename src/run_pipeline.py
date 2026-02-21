from src.model_spec import build_model
from src.fit_model import sample_model
from src.diagnostics import run_diagnostics
import pandas as pd
import numpy as np

pd.set_option("display.max_rows", 8)
pd.set_option("display.max_rows", 32)

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

    team_to_idx = (
        df[["team", "team_idx"]]
        .drop_duplicates()
        .set_index("team")["team_idx"]
        .to_dict()
    )

    results = run_diagnostics(trace, team_to_idx)

    print(results["summary"])
    print(results["team_effects"])

if __name__ == "__main__":
    main()