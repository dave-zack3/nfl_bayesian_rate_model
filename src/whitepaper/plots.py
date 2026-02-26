import matplotlib.pyplot as plt
import arviz as az
import numpy as np

def plot_rho_posterior(trace, label):
    az.plot_posterior(trace, var_names=["rho"])
    plt.title(f"Posterior Distribution of ρ — {label}")
    plt.tight_layout()
    plt.show()


def plot_offseason_shocks(trace, label):
    eta = trace.posterior["eta"].values.flatten()

    plt.hist(np.abs(eta), bins=30)
    plt.title(f"Distribution of Offseason Shock Magnitudes — {label}")
    plt.xlabel("|Shock|")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_team_strength(trace, team_index, team_name):
    theta = trace.posterior["theta"].mean(dim=("chain", "draw")).values

    plt.plot(theta[team_index])
    plt.title(f"Estimated Strength Trajectory — {team_name}")
    plt.xlabel("Week")
    plt.ylabel("Strength")
    plt.tight_layout()
    plt.show()


def plot_rolling_rmse(spread_results):
    plt.plot(spread_results["week"], spread_results["rmse"])
    plt.title("Rolling RMSE by Week")
    plt.xlabel("Week")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_rho_effect():
    np.random.seed(3)

    T = 100
    sigma = 1.0
    rhos = [0.7, 0.9, 0.98]

    plt.figure()

    for rho in rhos:
        theta = [0]
        for t in range(1, T):
            theta.append(rho * theta[-1] + np.random.normal(0, sigma))
        plt.plot(theta, label=f"ρ = {rho}")

    plt.xlabel("Week")
    plt.ylabel("Team Strength")
    plt.title("Effect of ρ on Team Strength Trajectories")
    plt.legend()
    plt.show()