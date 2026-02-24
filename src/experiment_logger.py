import json
from pathlib import Path
from datetime import datetime
import numpy as np


def convert_numpy(obj):
    """
    Recursively convert numpy types to native Python types
    so they can be JSON serialized.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def log_experiment(config, summary, results):

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    record = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "summary": summary,
        "results": results
    }

    record = convert_numpy(record)

    filename = log_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(filename, "w") as f:
        json.dump(record, f, indent=2)

    print(f"Experiment logged to {filename}")