import pandas as pd

def load_vegas_spreads(path="data/raw/vegas_spreads.csv"):
    vegas = pd.read_csv(path)
    return vegas[["game_id", "closing_spread"]]