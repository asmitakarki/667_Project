"""
This is for plotting the training curves that were downloaded from the LCC cluster.
Plot training curves from logged CSV files for agents. Manually change the paths as needed. 
"""

import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np

files = glob.glob("logs/SAC/monitor/*.csv")

dfs = []
for f in files:
    df = pd.read_csv(f, skiprows=1)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# sort by time
data = data.sort_values("t")

data["steps"] = data["l"].cumsum()

BIN_SIZE = 5000  # timesteps per bin 
data["bin"] = (data["steps"] // BIN_SIZE).astype(int)

binned = data.groupby("bin").agg(
    steps=("steps", "max"),
    mean_r=("r", "mean"),
    std_r=("r", "std"),
).reset_index()

# smooth the binned mean 
binned["mean_r_smooth"] = binned["mean_r"].rolling(10, min_periods=1).mean()

plt.figure(figsize=(9, 5))
plt.plot(binned["steps"], binned["mean_r_smooth"], label="SAC (binned + smoothed)")
plt.xlabel("Timesteps")
plt.ylabel("Episode Reward")
plt.title("TD3 Training Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("graphs/sac_training_curve.png", dpi=150)
plt.show()
