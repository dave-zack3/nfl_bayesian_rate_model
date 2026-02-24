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

    # Time index within each season
    model_df["time_idx"] = (
        model_df.groupby("season")["week"]
                .transform(lambda x: x.rank(method="dense").astype(int) - 1)
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
        "version": version,
        "years": years,
        "n_rows": len(model_df),
        "n_teams": model_df["team"].nunique(),
        "n_seasons": model_df["season"].nunique(),
        "n_weeks_per_season": (
            model_df.groupby("season")["time_idx"]
                    .nunique()
                    .to_dict()
        ),
        "created_at_utc": datetime.utcnow().isoformat()
    }

    meta_path = processed_dir / f"nfl_dynamic_ready_{version}_metadata.json"

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata written → {meta_path}")

    return final_path