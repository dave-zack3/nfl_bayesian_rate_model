import pymc as pm
import numpy as np
import pytensor.tensor as pt
from pytensor.scan import scan


def build_spread_model(
    team_idx,
    opp_idx,
    time_idx,
    season_id,
    is_season_start,
    spread,
    home_flag,
    n_teams,
    n_weeks,
    n_seasons,
    model_config
):

    with pm.Model() as model:

        # =================================================
        # RHO PRIOR
        # =================================================

        rho_cfg = model_config["rho_prior"]

        if rho_cfg["type"] == "beta":
            rho = pm.Beta("rho", rho_cfg["a"], rho_cfg["b"])
        elif rho_cfg["type"] == "uniform":
            rho = pm.Uniform("rho", 0, 1)
        else:
            raise ValueError("Unsupported rho prior")

        # =================================================
        # WEEKLY VOLATILITY
        # =================================================

        sigma_cfg = model_config["sigma_theta_prior"]

        if sigma_cfg["type"] == "halfnormal":
            sigma_theta = pm.HalfNormal(
                "sigma_theta",
                sigma_cfg["scale"]
            )
        else:
            raise ValueError("Unsupported sigma prior")

        # =================================================
        # OFFSEASON VOLATILITY
        # =================================================

        sigma_offseason = pm.HalfNormal(
            "sigma_offseason",
            model_config.get("sigma_offseason_scale", 0.3)
        )

        # Offseason shocks: one per team per season
        eta = pm.Normal(
            "eta",
            0,
            sigma_offseason,
            shape=(n_seasons, n_teams)
        )

        pm.Deterministic(
            "offseason_shock_magnitude",
            pt.mean(pt.abs(eta))
        )

        # =================================================
        # INITIAL TEAM STRENGTH
        # =================================================

        theta_init = pm.Normal(
            "theta_init",
            0,
            model_config.get("theta_init_scale", 0.3),
            shape=n_teams
        )

        # =================================================
        # WEEKLY INNOVATIONS (time-major for scan)
        # =================================================

        eps = pm.Normal(
            "eps",
            0,
            sigma_theta,
            shape=(n_weeks - 1, n_teams)
        )

        # Convert season metadata to tensors
        season_id_t = pt.as_tensor_variable(season_id)
        is_start_t = pt.as_tensor_variable(is_season_start)

        # We only need time 1...T-1 inside scan
        season_seq = season_id_t[1:]
        start_seq = is_start_t[1:]

        # =================================================
        # AR(1) + OFFSEASON RESET (SCAN VERSION)
        # =================================================

        def ar1_step(eps_t, season_t, start_t, theta_prev, rho, eta):

            # Standard AR(1)
            theta_t = rho * theta_prev + eps_t

            # Add offseason shock if start of season
            theta_t = theta_t + start_t * eta[season_t]

            return theta_t

        theta_seq, _ = scan(
            fn=ar1_step,
            sequences=[eps, season_seq, start_seq],
            outputs_info=theta_init,
            non_sequences=[rho, eta]
        )

        # Prepend initial state
        theta_full = pt.concatenate(
            [theta_init[None, :], theta_seq],
            axis=0
        )

        # shape → (n_teams, n_weeks)
        theta_raw = pm.Deterministic(
            "theta",
            theta_full.T
        )

        # Sum-to-zero per week
        theta = theta_raw - pt.mean(
            theta_raw,
            axis=0,
            keepdims=True
        )

        # =================================================
        # INTERPRETABLE COMPONENTS
        # =================================================
        pm.Deterministic("league_mean_strength", pt.mean(theta))

        pm.Deterministic(
            "avg_abs_strength",
            pt.mean(pt.abs(theta))
        )

        pm.Deterministic(
            "avg_weekly_volatility",
            sigma_theta
        )

        pm.Deterministic(
            "avg_offseason_volatility",
            sigma_offseason
        )

        # =================================================
        # HOME FIELD EFFECT
        # =================================================

        beta_home = pm.Normal(
            "beta_home",
            0,
            model_config.get("beta_home_scale", 3)
        )

        # =================================================
        # MEAN STRUCTURE
        # =================================================

        mu_spread = (
            theta[team_idx, time_idx]
            - theta[opp_idx, time_idx]
            + beta_home * home_flag
        )

        # =================================================
        # NOISE STRUCTURE
        # =================================================

        noise_type = model_config.get("noise_type", "heteroskedastic")

        if noise_type == "homoskedastic":

            sigma = pm.HalfNormal(
                "sigma0",
                model_config.get("sigma0_scale", 10)
            )

        elif noise_type == "heteroskedastic":

            sigma0 = pm.HalfNormal(
                "sigma0",
                model_config.get("sigma0_scale", 10)
            )

            alpha = pm.HalfNormal(
                "alpha",
                model_config.get("alpha_scale", 0.5)
            )

            sigma = sigma0 * (1 + alpha * pt.abs(mu_spread))

            pm.Deterministic(
                "avg_noise_scale",
                pt.mean(sigma)
            )

        else:
            raise ValueError("Unsupported noise type")

        # =================================================
        # OBSERVATION MODEL
        # =================================================

        pm.Normal(
            "spread_obs",
            mu=mu_spread,
            sigma=sigma,
            observed=spread
        )

    return model