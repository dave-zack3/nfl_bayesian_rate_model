import numpy as np
import pandas as pd


def expected_score(rating_a, rating_b):
    """
    Standard Elo logistic expectation.
    """
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def elo_update(rating, expected, actual, k=20):
    """
    Elo update step.
    """
    return rating + k * (actual - expected)


def run_elo_backtest(df, target_season, k=20, home_advantage=55):
    """
    Expanding-window Elo benchmark.

    - Initializes all teams at 1500
    - Updates sequentially through games
    - Produces spread prediction as rating difference
    """

    df = df.sort_values(["season", "time_idx", "game_id"]).copy()

    teams = df["team"].unique()
    ratings = {team: 1500 for team in teams}

    results = []

    season_df = df[df["season"] == target_season]
    weeks = sorted(season_df["time_idx"].unique())

    for week in weeks:

        week_games = season_df[season_df["time_idx"] == week]

        # Each game appears twice (home + away rows)
        home_rows = week_games[week_games["home_flag"] == 1]

        for _, row in home_rows.iterrows():

            home = row["team"]
            away = row["opponent"]

            home_rating = ratings[home] + home_advantage
            away_rating = ratings[away]

            # Elo predicted spread proxy
            predicted_spread = (home_rating - away_rating) / 25

            # Observed spread
            away_points = week_games[
                (week_games["team"] == away) &
                (week_games["game_id"] == row["game_id"])
            ]["points"].values[0]

            observed_spread = row["points"] - away_points

            # Win indicator
            actual = 1 if observed_spread > 0 else 0
            expected = expected_score(home_rating, away_rating)

            # Update ratings
            ratings[home] = elo_update(ratings[home], expected, actual, k)
            ratings[away] = elo_update(ratings[away], 1 - expected, 1 - actual, k)

            results.append({
                "week": week,
                "predicted_spread": predicted_spread,
                "observed_spread": observed_spread,
                "prob_home_win": expected
            })

    return pd.DataFrame(results)