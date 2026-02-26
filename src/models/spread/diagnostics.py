import numpy as np

def summarize_spread_structure(trace):
    posterior = trace.posterior

    def mean_sd(name):
        values = posterior[name].values.flatten()
        return {
            "mean": float(np.mean(values)),
            "sd": float(np.std(values))
        }

    summary = {}

    # Core parameters
    if "rho" in posterior:
        summary["rho"] = mean_sd("rho")

    if "sigma_theta" in posterior:
        summary["sigma_weekly"] = mean_sd("sigma_theta")

    if "sigma_offseason" in posterior:
        summary["sigma_offseason"] = mean_sd("sigma_offseason")

    if "beta_home" in posterior:
        summary["beta_home"] = mean_sd("beta_home")

    # Noise
    if "sigma0" in posterior:
        summary["sigma0"] = mean_sd("sigma0")

    if "alpha" in posterior:
        summary["alpha"] = mean_sd("alpha")

    # League-level interpretability
    if "theta" in posterior:
        theta = posterior["theta"].values
        summary["avg_abs_strength"] = {
            "mean": float(np.mean(np.abs(theta)))
        }

    if "eta" in posterior:
        eta = posterior["eta"].values
        summary["avg_offseason_shock"] = {
            "mean": float(np.mean(np.abs(eta)))
        }

    return summary