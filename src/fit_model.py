import pymc as pm

def sample_model(model):
    with model:
        trace = pm.sample(
            1000,
            tune=1000,
            target_accept=0.9,
            return_inferencedata=True
        )
    return trace