import numpy as np
from src.models.spread.spread_model_spec import build_spread_model


def build_spread_model_from_df(df, model_config):

    # -----------------------------------------
    # Extract indexing arrays
    # -----------------------------------------

    team_idx = df["home_team_idx"].values
    opp_idx = df["away_team_idx"].values
    time_idx = df["time_idx"].values

    season_id = df["season_id"].values
    is_season_start = df["is_season_start"].values
    n_seasons = int(df["season_id"].max()) + 1

    spread = df["spread"].values

    # Since dataset is already home-perspective
    home_flag = np.ones_like(spread)

    # -----------------------------------------
    # Compute structural dimensions
    # -----------------------------------------

    n_teams = int(
        max(
            df["home_team_idx"].max(),
            df["away_team_idx"].max()
        )
    ) + 1

    n_weeks = int(df["time_idx"].max()) + 1

    # -----------------------------------------
    # Build model
    # -----------------------------------------

    return build_spread_model(
        team_idx=team_idx,
        opp_idx=opp_idx,
        time_idx=time_idx,
        season_id=season_id,
        is_season_start=is_season_start,
        spread=spread,
        home_flag=home_flag,
        n_teams=n_teams,
        n_weeks=n_weeks,
        n_seasons=n_seasons,
        model_config=model_config
    )