"""
Plot combined training curves (PPO + SAC + TD3) from LCC-downloaded SB3 Monitor CSVs.
Uses: steps = cumsum(l), binning, mean/std, rolling smoothing.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ALGOS = ["PPO", "SAC", "TD3"]
LOG_ROOT = "logs"          # expects logs/<ALGO>/monitor/*.csv
OUT_DIR = "graphs"
OUT_FILE = os.path.join(OUT_DIR, "combined_training_curves.png")

BIN_SIZE = 5000            # timesteps per bin
SMOOTH_WINDOW = 10         # rolling window on binned mean

def load_and_bin(algo: str):
    files = glob.glob(os.path.join(LOG_ROOT, algo, "monitor", "*.csv"))
    if len(files) == 0:
        print(f"[WARN] No files found for {algo}: {LOG_ROOT}/{algo}/monitor/*.csv")
        return None

    dfs = []
    for f in files:
        df = pd.read_csv(f, skiprows=1)
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True).sort_values("t")

    data["steps"] = data["l"].cumsum()
    data["bin"] = (data["steps"] // BIN_SIZE).astype(int)

    binned = data.groupby("bin").agg(
        steps=("steps", "max"),
        mean_r=("r", "mean"),
        std_r=("r", "std"),
    ).reset_index()

    binned["mean_r_smooth"] = binned["mean_r"].rolling(SMOOTH_WINDOW, min_periods=1).mean()
    return binned

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    plt.figure(figsize=(9, 5))

    for algo in ALGOS:
        binned = load_and_bin(algo)
        if binned is None:
            continue

        plt.plot(
            binned["steps"],
            binned["mean_r_smooth"],
            label=f"{algo} (binned+smoothed)"
        )

    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward")
    plt.title("Training Curves: PPO vs SAC vs TD3")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=150)
    plt.show()
    print(f"Saved: {OUT_FILE}")

if __name__ == "__main__":
    main()