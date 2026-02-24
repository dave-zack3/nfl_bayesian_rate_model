from src.model_spec import build_model

def build_model_from_df(df):

    team_idx = df["team_idx"].values
    opp_idx = df["opponent_idx"].values
    time_idx = df["time_idx"].values
    points = df["points"].values
    drives = df["drives"].values
    home_flag = df["home_flag"].values

    n_teams = df["team_idx"].nunique()
    n_weeks = df["time_idx"].nunique()

    return build_model(
        team_idx,
        opp_idx,
        time_idx,
        points,
        drives,
        home_flag,
        n_teams,
        n_weeks
    )