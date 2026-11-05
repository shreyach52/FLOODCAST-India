import pandas as pd
import numpy as np

d = pd.read_csv("data/interim/baseline_predictions.csv.gz")

print("\nFINAL MODEL EVALUATION")
print("=" * 115)
print(f"{'Horizon':<9} {'Split':<19} {'Model':<13} {'N':>8} {'RMSE':>10} {'Extreme RMSE':>14} {'MAE':>10} {'R2':>9} {'Extreme N':>11}")
print("-" * 115)

for (h, s, m), g in d.groupby(["horizon", "split", "model"]):

    g = g.dropna(subset=["actual", "predicted", "high_flow_threshold"])

    y = g["actual"].to_numpy()
    p = g["predicted"].to_numpy()

    errors = y - p

    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))

    denominator = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum(errors ** 2) / denominator if denominator > 0 else np.nan

    extreme = y >= g["high_flow_threshold"].to_numpy()

    extreme_rmse = (
        np.sqrt(np.mean(errors[extreme] ** 2))
        if extreme.any() else np.nan
    )

    print(
        f"{h}d       {s:<19} {m:<13} {len(g):>8} "
        f"{rmse:>10.3f} {extreme_rmse:>14.3f} "
        f"{mae:>10.3f} {r2:>9.4f} {extreme.sum():>11}"
    )

print("=" * 115)
