import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import nfl_data_py as nfl

def detect_available_seasons(min_year=2009):
    import nfl_data_py as nfl
    years = nfl.import_schedules(range(min_year, 2100))["season"].unique()
    return sorted(years)

def build_season_dataset(year, output_dir, force_rebuild=False):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    season_path = output_dir / f"nfl_team_game_{year}.csv"

    if season_path.exists() and not force_rebuild:
        print(f"Season {year} already cached.")
        return season_path

    print(f"Building season dataset for {year}...")

    pbp_df = nfl.import_pbp_data([year])
    pbp_df = pbp_df[pbp_df["season_type"] == "REG"]

    pbp_df = pbp_df[
        ["game_id", "season", "week",
         "posteam", "home_team", "away_team",
         "total_home_score", "total_away_score",
         "drive"]
    ]

    drives = (
        pbp_df.groupby(["game_id", "posteam"])["drive"]
              .nunique()
              .reset_index(name="drives")
    )

    final_scores = (
        pbp_df.groupby("game_id")
              .tail(1)
              [["game_id", "season", "week",
                "home_team", "away_team",
                "total_home_score", "total_away_score"]]
    )

    home_df = final_scores[
        ["game_id", "season", "week",
         "home_team", "away_team",
         "total_home_score"]
    ].copy()

    home_df.columns = ["game_id", "season", "week",
                       "team", "opponent", "points"]
    home_df["home_flag"] = 1

    away_df = final_scores[
        ["game_id", "season", "week",
         "away_team", "home_team",
         "total_away_score"]
    ].copy()

    away_df.columns = ["game_id", "season", "week",
                       "team", "opponent", "points"]
    away_df["home_flag"] = 0

    game_df = pd.concat([home_df, away_df], ignore_index=True)

    model_df = pd.merge(
        left=game_df,
        right=drives,
        left_on=["game_id", "team"],
        right_on=["game_id", "posteam"],
        how="left"
    )

    model_df = model_df.sort_values(["season", "week", "game_id"])

    model_df.to_csv(season_path, index=False)
    print(f"Saved season {year} → {season_path}")

    return season_path

def build_multi_season_dataset(
    years,
    processed_dir="data/processed",
    version="v1",
    force_rebuild=False
):

    processed_dir = Path(processed_dir)
    season_dir = processed_dir / "seasons"
    season_dir.mkdir(parents=True, exist_ok=True)

    season_paths = []

    for year in years:
        path = build_season_dataset(
            year,
            season_dir,
            force_rebuild=force_rebuild
        )
        season_paths.append(path)

    # -----------------------------
    # Aggregate seasons
    # -----------------------------

    print("Aggregating seasons...")
    dfs = [pd.read_csv(p) for p in season_paths]
    model_df = pd.concat(dfs, ignore_index=True)

    # Stable team index across seasons
    teams = sorted(model_df["team"].unique())
    team_to_idx = {team: i for i, team in enumerate(teams)}

    model_df["team_idx"] = model_df["team"].map(team_to_idx)
    model_df["opponent_idx"] = model_df["opponent"].map(team_to_idx)

    # ---------------------------------------------------
    # GLOBAL sequential time index across all seasons
    # ---------------------------------------------------

    model_df = model_df.sort_values(["season", "week", "game_id"])

    # Create global week counter
    season_week = (
        model_df[["season", "week"]]
        .drop_duplicates()
        .sort_values(["season", "week"])
        .reset_index(drop=True)
    )

    season_week["global_time_idx"] = range(len(season_week))

    model_df = model_df.merge(
        season_week,
        on=["season", "week"],
        how="left"
    )

    model_df["time_idx"] = model_df["global_time_idx"]

    model_df.drop(columns=["global_time_idx"], inplace=True)

    # ---------------------------------------------------
    # Season ID (integer 0,1,2,...)
    # ---------------------------------------------------

    season_lookup = (
        model_df[["season"]]
        .drop_duplicates()
        .sort_values("season")
        .reset_index(drop=True)
    )

    season_lookup["season_id"] = range(len(season_lookup))

    model_df = model_df.merge(
        season_lookup,
        on="season",
        how="left"
    )

    # ---------------------------------------------------
    # Flag season start (first week of each season)
    # ---------------------------------------------------

    first_week_lookup = (
        model_df.groupby("season")["time_idx"]
        .min()
        .reset_index()
    )

    first_week_lookup["is_season_start"] = 1

    model_df = model_df.merge(
        first_week_lookup,
        on=["season", "time_idx"],
        how="left"
    )

    model_df["is_season_start"] = (
        model_df["is_season_start"]
        .fillna(0)
        .astype(int)
    )

    # -----------------------------
    # Write final dataset
    # -----------------------------

    final_path = processed_dir / f"nfl_dynamic_ready_{version}.csv"
    model_df.to_csv(final_path, index=False)

    print(f"Final dataset written → {final_path}")

    # -----------------------------
    # Write metadata JSON
    # -----------------------------

    metadata = {
        "version": str(version),
        "years": [int(y) for y in years],
        "n_rows": int(len(model_df)),
        "n_teams": int(model_df["team"].nunique()),
        "n_seasons": int(model_df["season"].nunique()),
        "n_weeks_per_season": {
            int(k): int(v)
            for k, v in (
                model_df.groupby("season")["time_idx"]
                        .nunique()
                        .to_dict()
                        .items()
            )
        },
        "created_at_utc": datetime.utcnow().isoformat()
    }

    meta_path = processed_dir / f"nfl_dynamic_ready_{version}_metadata.json"

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata written → {meta_path}")

    return final_path

def build_game_level_spread_dataset(df):

    """
    Converts team-game level dataframe into
    game-level spread dataset (one row per game).
    """

    # Keep only home rows (one per game)
    home_df = df[df["home_flag"] == 1].copy()

    # Identify away rows
    away_df = df[df["home_flag"] == 0].copy()

    # Merge home and away on game_id
    merged = home_df.merge(
        away_df,
        on="game_id",
        suffixes=("_home", "_away")
    )

    # Compute spread (home - away)
    merged["spread"] = (
        merged["points_home"]
        - merged["points_away"]
    )

    # Build clean dataset
    spread_df = merged[[
        "game_id",
        "season_home",
        "season_id_home",
        "time_idx_home",
        "is_season_start_home",
        "team_idx_home",
        "team_idx_away",
        "spread"
    ]].copy()

    spread_df.columns = [
        "game_id",
        "season",
        "season_id",
        "time_idx",
        "is_season_start",
        "home_team_idx",
        "away_team_idx",
        "spread"
    ]

    spread_df["home_flag"] = 1  # always home perspective

    return spread_df