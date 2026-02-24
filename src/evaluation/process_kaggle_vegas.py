import pandas as pd
from pathlib import Path

def create_vegas_spread_file(
    kaggle_path="data/raw/kaggle_nfl_odds.csv",
    model_data_path="data/processed/nfl_dynamic_ready_v1.csv",
    output_path="data/raw/vegas_spreads.csv"
):

    kaggle = pd.read_csv(kaggle_path)
    model_df = pd.read_csv(model_data_path)

    # ----------------------------------------
    # Standardize column names
    # ----------------------------------------

    kaggle = kaggle.rename(columns={
        "team_home": "home_team",
        "team_away": "away_team",
        "schedule_date": "date",
        "spread_favorite": "closing_spread"
    })

    # ----------------------------------------
    # Convert date to season/year
    # ----------------------------------------

    kaggle["date"] = pd.to_datetime(kaggle["date"])
    kaggle["season"] = kaggle["date"].dt.year

    # Adjust for Jan playoff games belonging to previous season
    kaggle.loc[
        kaggle["date"].dt.month <= 2,
        "season"
    ] -= 1

    # ----------------------------------------
    # Map team names to your model naming
    # ----------------------------------------
    team_map = {
        "Arizona Cardinals": "ARI",
        "Atlanta Falcons": "ATL",
        "Baltimore Ravens": "BAL",
        "Buffalo Bills": "BUF",
        "Carolina Panthers": "CAR",
        "Chicago Bears": "CHI",
        "Cincinnati Bengals": "CIN",
        "Cleveland Browns": "CLE",
        "Dallas Cowboys": "DAL",
        "Denver Broncos": "DEN",
        "Detroit Lions": "DET",
        "Green Bay Packers": "GB",
        "Houston Texans": "HOU",
        "Indianapolis Colts": "IND",
        "Jacksonville Jaguars": "JAX",
        "Kansas City Chiefs": "KC",
        "Las Vegas Raiders": "LV",
        "Los Angeles Chargers": "LAC",
        "Los Angeles Rams": "LAR",
        "Miami Dolphins": "MIA",
        "Minnesota Vikings": "MIN",
        "New England Patriots": "NE",
        "New Orleans Saints": "NO",
        "New York Giants": "NYG",
        "New York Jets": "NYJ",
        "Philadelphia Eagles": "PHI",
        "Pittsburgh Steelers": "PIT",
        "San Francisco 49ers": "SF",
        "Seattle Seahawks": "SEA",
        "Tampa Bay Buccaneers": "TB",
        "Tennessee Titans": "TEN",
        "Washington Commanders": "WAS",
        "Houston Oilers": "TEN",
        "Oakland Raiders": "LV",
        "San Diego Chargers": "LAC",
        "St. Louis Rams": "LAR",
        "Washington Redskins": "WAS",
        "Washington Football Team": "WAS"
    }

    kaggle["home_team"] = kaggle["home_team"].replace(team_map)
    kaggle["away_team"] = kaggle["away_team"].replace(team_map)

    if kaggle["home_team"].isna().any():
        missing = kaggle[kaggle["home_team"].isna()]["home_team"].unique()
        raise ValueError(f"Unmapped home teams found: {missing}")

    if kaggle["away_team"].isna().any():
        missing = kaggle[kaggle["away_team"].isna()]["away_team"].unique()
        raise ValueError(f"Unmapped away teams found: {missing}")

    # ----------------------------------------
    # Merge with model dataset to obtain game_id
    # ----------------------------------------

    # Keep only home rows in model data
    model_home = model_df[model_df["home_flag"] == 1][
        ["game_id", "season", "week", "team", "opponent"]
    ].copy()

    model_home = model_home.rename(columns={
        "team": "home_team",
        "opponent": "away_team"
    })

    merged = model_home.merge(
        kaggle,
        on=["season", "home_team", "away_team"],
        how="left"
    )

    # ----------------------------------------
    # Validate merge
    # ----------------------------------------

    missing = merged["closing_spread"].isna().sum()
    if missing > 0:
        print(f"WARNING: {missing} games missing spreads")

    vegas_final = merged[["game_id", "closing_spread"]]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    vegas_final.to_csv(output_path, index=False)

    print(f"Vegas spread file written to {output_path}")