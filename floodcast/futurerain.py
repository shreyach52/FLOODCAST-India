"""Value-of-future-rain experiment: building blocks (pandas/numpy only, unit-tested).

DIAGNOSTIC ONLY. The `oracle_*` features contain rain that fell AFTER the forecast was issued. They show
how much a rainfall forecast COULD add; they must never reach the deployed model, the dashboard or the
submission. Degrading the oracle to a given forecast skill:
    forecast_z = rho * true_z + sqrt(1 - rho^2) * noise_z      (z = standardised log1p rain accumulation)
so rho is the correlation between forecast and observed rain accumulation over the next h days
(rho=1: perfect knowledge, rho=0: no information).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ORACLE_PREFIX = "oracle_"
H = (1, 3, 7)
DIS_LAGS = (1, 2, 3, 5, 7, 14)
DIS_WINDOWS = (3, 7, 14, 30)
DIS_FEATURES = ([f"lag{k}" for k in DIS_LAGS] + [f"mean{w}" for w in DIS_WINDOWS] +
                ["max7", "diff1", "diff3", "doy_sin", "doy_cos"])


def oracle_cols(h: int) -> list:
    """Future-rain feature names for horizon h (h=1: sum only, max would be identical)."""
    return [f"{ORACLE_PREFIX}sum{h}"] + ([f"{ORACLE_PREFIX}max{h}"] if h > 1 else [])


def future_rain(w: pd.DataFrame, horizons=H) -> pd.DataFrame:
    """Observed rain over days t+1..t+h for every station-day t (NaN if any of those days is missing)."""
    parts = []
    for sid, g in w.groupby("station_id"):
        g = g.sort_values("date").drop_duplicates("date").set_index("date")
        idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
        r = g["rain_mm"].reindex(idx)
        f = pd.DataFrame(index=idx)
        for h in horizons:
            f[f"{ORACLE_PREFIX}sum{h}"] = r.rolling(h, min_periods=h).sum().shift(-h)
            if h > 1:
                f[f"{ORACLE_PREFIX}max{h}"] = r.rolling(h, min_periods=h).max().shift(-h)
        f.insert(0, "station_id", sid)
        f.index.name = "date"
        parts.append(f.reset_index())
    return pd.concat(parts, ignore_index=True)


def discharge_frame(g: pd.DataFrame, horizons=H) -> pd.DataFrame:
    """Past-only discharge features for one station (+ targets y{h} = discharge h days ahead)."""
    g = g.sort_values("date").drop_duplicates("date").set_index("date")
    idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
    q = g["discharge"].reindex(idx)
    lq = np.log1p(q)
    f = pd.DataFrame(index=idx)
    f["q0"] = q
    for k in DIS_LAGS:
        f[f"lag{k}"] = lq.shift(k)
    for w in DIS_WINDOWS:
        f[f"mean{w}"] = lq.rolling(w, min_periods=max(2, w // 2)).mean()
    f["max7"] = lq.rolling(7, min_periods=4).max()
    f["diff1"] = lq - lq.shift(1)
    f["diff3"] = lq - lq.shift(3)
    doy = idx.dayofyear.to_numpy()
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    for h in horizons:
        f[f"y{h}"] = q.shift(-h)
    f.insert(0, "station_id", g["station_id"].iloc[0] if "station_id" in g else None)
    f.index.name = "date"
    return f.reset_index()


def zscore_log(df: pd.DataFrame, cols, ref_end: dict) -> pd.DataFrame:
    """Standardise log1p(col) per station using only rows dated <= that station's reference end."""
    out = df[["station_id", "date"]].copy()
    limit = df["station_id"].map(ref_end)
    for c in cols:
        lg = np.log1p(df[c])
        ref = lg.where(df["date"] <= limit)
        mu = ref.groupby(df["station_id"]).transform("mean")
        sd = ref.groupby(df["station_id"]).transform("std").replace(0, np.nan)
        out[c] = (lg - mu) / sd
    return out


def degrade(z: np.ndarray, rho: float, eps: np.ndarray) -> np.ndarray:
    """Forecast of skill rho from true z and a FIXED noise array eps (same noise for every rho, so
    the skill levels are nested and differences between them are not noise-sampling artefacts)."""
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    return rho * z + np.sqrt(1.0 - rho ** 2) * eps
