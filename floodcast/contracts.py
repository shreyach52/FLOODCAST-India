"""
FROZEN INTERFACE CONTRACTS  (v1)

Everyone imports names from this file. Change it ONLY by team agreement,
and tell everybody when you do. This is what lets each person work alone.

NOTE vs. the earlier draft: the target column is now called `discharge`
(not `discharge_cfs`) and the canonical unit is m3/s, because the public
CWC data is used. If a competition file in cfs shows up later, convert at
the edges with cfs_to_m3s / m3s_to_cfs. Never mix units inside the pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---- canonical column names -------------------------------------------------
STATION = "station_id"
DATE = "date"
TARGET = "discharge"
UNIT = "m3s"                      # canonical unit used everywhere inside the project
CANON_REQUIRED = [STATION, DATE, TARGET]
CANON_OPTIONAL = ["station_name", "river", "state", "lat", "lon"]

HORIZONS = (1, 3, 7)              # forecast days ahead
RISK_LEVELS = ["Normal", "Elevated", "High", "Extreme"]

# Every forecasting model must return a DataFrame with exactly these columns
FORECAST_COLS = ["station_id", "issue_date", "horizon", "q_pred", "q_low", "q_high"]

CFS_PER_M3S = 35.3146667          # 1 m3/s = 35.3147 cfs


def m3s_to_cfs(x):
    return x * CFS_PER_M3S


def cfs_to_m3s(x):
    return x / CFS_PER_M3S


# ---- validators -------------------------------------------------------------
def validate_canonical_df(df: pd.DataFrame) -> None:
    """The table every later step reads: one row per (station, day)."""
    missing = set(CANON_REQUIRED) - set(df.columns)
    assert not missing, f"Missing columns: {missing}"
    assert pd.api.types.is_datetime64_any_dtype(df[DATE]), "date must be datetime64"
    assert df[CANON_REQUIRED].notna().all().all(), "NaN in required columns"
    assert (df[TARGET] >= 0).all(), "Negative discharge"
    assert not df.duplicated([STATION, DATE]).any(), "Duplicate (station, date) rows"


def validate_forecast_df(df: pd.DataFrame) -> None:
    missing = set(FORECAST_COLS) - set(df.columns)
    assert not missing, f"Missing columns: {missing}"
    assert df["horizon"].isin(HORIZONS).all(), "Bad horizon value"
    vals = df[["q_pred", "q_low", "q_high"]].to_numpy(dtype=float)
    assert np.isfinite(vals).all(), "NaN or inf in predictions"
    assert (df["q_low"] >= 0).all(), "Negative discharge"
    assert (df["q_low"] <= df["q_pred"]).all(), "q_low above q_pred"
    assert (df["q_pred"] <= df["q_high"]).all(), "q_pred above q_high"


# ---- splits -----------------------------------------------------------------
def make_splits(df, date_col=DATE, station_col=STATION,
                val_frac=0.15, test_frac=0.15,
                holdout_station_frac=0.10, seed=42):
    """Chronological split on 'seen' stations plus held-out ('ungauged') stations.

    Returns train, val, test, ungauged:
      - train/val/test: stations seen in training, split by time
      - ungauged: stations never used for training, test period only
    """
    rng = np.random.default_rng(seed)
    stations = np.sort(df[station_col].unique())
    n_hold = max(1, int(len(stations) * holdout_station_frac))
    held = set(rng.choice(stations, size=n_hold, replace=False))

    seen = df[~df[station_col].isin(held)]
    dates = np.sort(seen[date_col].unique())
    t_val = dates[int(len(dates) * (1 - val_frac - test_frac))]
    t_test = dates[int(len(dates) * (1 - test_frac))]

    train = seen[seen[date_col] < t_val]
    val = seen[(seen[date_col] >= t_val) & (seen[date_col] < t_test)]
    test = seen[seen[date_col] >= t_test]
    ungauged = df[df[station_col].isin(held) & (df[date_col] >= t_test)]
    return train, val, test, ungauged
