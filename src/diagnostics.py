import arviz as az

def run_diagnostics(trace):
    summary = az.summary(trace)
    return summary