"""
Saves ESOL and FreeSolv results from terminal output into results_dmpnn_regr.json
Run this ONCE before running dmpnn_regr.py
"""
import json, os

SAVE_PATH = "results_dmpnn_regr.json"

results = {
    "ESOL": {
        "mean": 1.4638, "std": 0.0279, "metric": "rmse",
        "seeds": [1.4996, 1.4602, 1.4316],
        "best_params": {"hidden": 128, "num_layers": 2, "dropout": 0.28466566117599995,
                        "lr": 0.0009239150319627245, "weight_decay": 4.138040112561016e-05},
        "time_min": 14.8
    },
    "FreeSolv": {
        "mean": 6.3362, "std": 0.4553, "metric": "rmse",
        "seeds": [6.8008, 5.7178, 6.4899],
        "best_params": {"hidden": 256, "num_layers": 3, "dropout": 0.1576145845201955,
                        "lr": 0.0005766759829490925, "weight_decay": 1.8408446202037763e-06},
        "time_min": 2.9
    }
}

with open(SAVE_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved ESOL + FreeSolv → {SAVE_PATH}")
print("Now run: python dmpnn_regr.py")
