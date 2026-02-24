import pymc as pm

def sample_model(model,
                 init_trace=None,
                 chains=2,
                 cores=2,
                 draws=300,
                 tune=300,
                 target_accept=0.95,
                 max_treedepth=15):

    with model:

        if init_trace is not None:
            start = {
                var: init_trace.posterior[var].mean(dim=("chain", "draw")).values
                for var in init_trace.posterior.data_vars
            }
        else:
            start = None

        trace = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=target_accept,
            initvals=start,
            chains=chains,
            cores=cores,
            nuts={"max_treedepth": max_treedepth}
        )

    return trace