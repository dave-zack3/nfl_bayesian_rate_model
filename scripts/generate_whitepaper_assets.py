import pandas as pd
from src.run_pipeline import main, CONFIG
from src.whitepaper.plots import plot_rho_posterior
from src.models.spread.spread_model_wrapper import build_spread_model_from_df
from src.fit_model import sample_model
from src.data_loader import build_game_level_spread_dataset

# 1️⃣ Run pipeline to get performance results
results = main(CONFIG)

# 2️⃣ Load full dataset again for structural plotting
df = pd.read_csv("data/processed/nfl_dynamic_ready_v1.csv")
spread_df = build_game_level_spread_dataset(df)

# 3️⃣ Choose experiment (e.g., Beta(8,2))
experiment = {
    "name": "Beta(8,2)",
    "rho_prior": {"type": "beta", "a": 8, "b": 2},
    "sigma_theta_prior": {"type": "halfnormal", "scale": 0.2}
}

# 4️⃣ Fit full structural model
full_model = build_spread_model_from_df(spread_df, experiment)
full_trace = sample_model(full_model)

# 5️⃣ Plot posterior of rho
plot_rho_posterior(full_trace, experiment["name"])