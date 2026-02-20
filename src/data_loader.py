import pandas as pd
import nfl_data_py as nfl

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

pbp_df = pd.read_csv("../data/raw/pbp_2021.csv")

pbp_df = pbp_df[pbp_df["season_type"]=="REG"]

drives = (
    pbp_df.groupby(["game_id","posteam"])["drive"]
          .nunique()
          .reset_index(name = "drives")
)

final_scores = (
    pbp_df.groupby("game_id")
          .tail(1)
          [["game_id", "home_team", "away_team",
            "total_home_score", "total_away_score"]]
)

home_df = final_scores[["game_id",
                        "home_team",
                        "away_team",
                        "total_home_score"
]].copy()

home_df.columns = ["game_id", "team", "opponent", "points"]
home_df["home_flag"] = 1

away_df = final_scores[["game_id",
                        "away_team",
                        "home_team",
                        "total_away_score"
]].copy()

away_df.columns = ["game_id", "team", "opponent", "points"]
away_df["home_flag"] = 0

game_df = pd.concat([home_df,away_df], ignore_index=True)

model_df = pd.merge(left=game_df, 
                    right=drives, 
                    left_on=["game_id", "team"],
                    right_on=["game_id","posteam"],
                    how="left")

model_df.dropna()

teams = sorted(model_df["team"].unique())
team_to_idx = {team: i for i, team in enumerate(teams)}

model_df["team_idx"] = model_df["team"].map(team_to_idx)
model_df["opponent_idx"] = model_df["opponent"].map(team_to_idx)

def validate_team_game_df(df):

    # Row count check
    if df.shape[0] != 544:
        raise ValueError(f"Expected 544 rows, got {df.shape[0]}")

    # Unique teams
    if df["team"].nunique() != 32:
        raise ValueError("Expected 32 unique teams")

    # Unique games
    if df["game_id"].nunique() != 272:
        raise ValueError("Expected 272 unique games")

    # Duplicate check
    if df.duplicated(subset=["game_id", "team"]).sum() != 0:
        raise ValueError("Duplicate game/team rows found")

    # Null check
    if df.isna().sum().sum() != 0:
        raise ValueError("Null values detected")

    # Logical checks
    if (df["team"] == df["opponent"]).any():
        raise ValueError("Team playing itself detected")

    if (df["drives"] <= 0).any():
        raise ValueError("Non-positive drives detected")

    if (df["points"] < 0).any():
        raise ValueError("Negative points detected")

    if df["team_idx"].min() != 0 or df["team_idx"].max() != 31:
        raise ValueError("team_idx out of expected range")

    if df["opponent_idx"].min() != 0 or df["opponent_idx"].max() != 31:
        raise ValueError("opponent_idx out of expected range")

    print("Validation passed.")

validate_team_game_df(model_df)
model_df.to_csv("../data/processed/nfl_2021_team_game.csv", index=False)

