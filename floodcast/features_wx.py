"""Rainfall / soil-moisture features (pandas/numpy only, unit-tested).

Input  : weather table with columns station_id, date, rain_mm, soil1 (from 06b_extract_station_weather.py).
Output : one row per station-day with WX_FEATURES.
Leakage rule: every feature on date t uses weather dated <= t only (right-aligned windows).
ERA5-Land rain for date t is the 24 h ending 05:30 IST of day t, so it is known before day t's discharge reading.
NOTE: no FUTURE rain is used. At 3/7-day horizons the model therefore sees what already fell, not what will fall.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RAIN_WINDOWS = (3, 7, 14, 30)
WX_FEATURES = (["rain1"] + [f"rain{w}" for w in RAIN_WINDOWS] +
               ["rain_max3", "api", "soil1", "soil1_d3", "rain7_rel", "rain30_rel", "wet_days14"])


def _station_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").drop_duplicates("date").set_index("date")
    idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
    r = g["rain_mm"].reindex(idx)
    s = g["soil1"].reindex(idx)
    f = pd.DataFrame(index=idx)
    f["rain1"] = r
    for w in RAIN_WINDOWS:
        f[f"rain{w}"] = r.rolling(w, min_periods=w).sum()
    f["rain_max3"] = r.rolling(3, min_periods=3).max()
    # antecedent precipitation index: exponentially decaying wetness memory (decay 0.9/day)
    f["api"] = r.fillna(0).ewm(alpha=0.1, adjust=False).mean() * 10.0
    f["soil1"] = s
    f["soil1_d3"] = s - s.shift(3)
    f["wet_days14"] = (r > 1.0).astype(float).where(r.notna()).rolling(14, min_periods=14).sum()
    return f


def weather_features(w: pd.DataFrame, train_end, lag_days: int = 0) -> pd.DataFrame:
    """Per-station features + station-relative rain (divided by the station's TRAIN-period mean, so scale is
    comparable across stations without peeking at later years)."""
    train_end = pd.Timestamp(train_end)
    parts = []
    for sid, g in w.groupby("station_id"):
        f = _station_features(g)
        tr = f.loc[f.index <= train_end]
        m7 = tr["rain7"].mean() if tr["rain7"].notna().any() else np.nan
        m30 = tr["rain30"].mean() if tr["rain30"].notna().any() else np.nan
        f["rain7_rel"] = f["rain7"] / (m7 + 1.0)
        f["rain30_rel"] = f["rain30"] / (m30 + 1.0)
        if lag_days < 0:
            raise ValueError("lag_days must be >= 0")
        if lag_days > 0:
            f[WX_FEATURES] = f[WX_FEATURES].shift(lag_days)
        f.insert(0, "station_id", sid)
        f.index.name = "date"
        parts.append(f.reset_index())
    return pd.concat(parts, ignore_index=True)[["station_id", "date"] + list(WX_FEATURES)]
