
---

# 📁 2️⃣ `src/README.md`

```markdown
# Source Code Overview

This directory contains all modeling, evaluation, and pipeline logic.

---

## Core Modules

### model_spec.py
Defines the dynamic Bayesian Negative Binomial model.

### fit_model.py
Wraps PyMC sampling configuration.

### model_wrapper.py
Extracts arrays from dataframe and builds model.

### run_pipeline.py
Main execution entrypoint.

---

## Evaluation

Located in `src/evaluation/`.

Includes:

- Rolling backtest
- Spread metrics
- Elo benchmark
- Vegas comparison

---

## Experiment Logging

experiment_logger.py logs configuration and results to JSON for reproducibility.