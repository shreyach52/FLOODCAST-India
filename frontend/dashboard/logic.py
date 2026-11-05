"""Pure helpers for the dashboard API. Read-only: no model code, nothing is predicted or invented here."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULTS = {
    "train_end": "2018-12-31",      # same train cut-off the backend uses for its high-flow threshold
    "high_quantile": 0.95,          # backend definition: station's train-period 95th percentile
    "elevated_quantile": 0.90,      # CONFIGURED indicative level (change in configs/frontend.json)
    "fresh_days": 7,                # a station counts as "reporting" if its last value is this recent
    "min_train_obs": 365,           # below this, no threshold is shown (not enough history)
    "forecast_file": "data/processed/forecasts.csv.gz",
    "weather_file": "data/interim/weather_daily.csv.gz",
    "metrics_files": ["data/interim/baseline_metrics.csv"],
}
RISKS = ("Normal", "Elevated", "High")


def load_cfg(path: Path) -> dict:
    cfg = dict(DEFAULTS)
    if Path(path).exists():
        cfg.update(json.loads(Path(path).read_text(encoding="utf-8")))
    return cfg


def num(v, nd=3):
    """float or None (JSON has no NaN)."""
    return None if v is None or pd.isna(v) or np.isinf(v) else round(float(v), nd)


def txt(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


def numlist(a, nd=3):
    return [num(v, nd) for v in a]


def thresholds(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    tr = df[df["date"] <= pd.Timestamp(cfg["train_end"])]
    g = tr.groupby("station_id")["discharge"]
    out = pd.DataFrame({"n_train": g.size(), "elevated": g.quantile(cfg["elevated_quantile"]),
                        "high": g.quantile(cfg["high_quantile"])})
    out.loc[out.n_train < cfg["min_train_obs"], ["elevated", "high"]] = np.nan
    return out


def classify(q, elevated, high):
    if any(v is None or pd.isna(v) for v in (q, elevated, high)):
        return None
    return "High" if q >= high else "Elevated" if q >= elevated else "Normal"


def station_table(df: pd.DataFrame, thr: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    last = df.sort_values("date").groupby("station_id").tail(1).set_index("station_id")
    info_cols = [c for c in ("station_name", "river", "state", "lat", "lon") if c in df.columns]
    t = last[["date", "discharge"] + info_cols].rename(columns={"date": "last_date", "discharge": "last_q"})
    t = t.join(thr, how="left")
    data_max = df["date"].max()
    t["fresh"] = t["last_date"] >= data_max - pd.Timedelta(days=int(cfg["fresh_days"]))
    t["risk"] = [classify(q, e, h) if f else None
                 for q, e, h, f in zip(t.last_q, t.elevated, t.high, t.fresh)]
    return t.sort_index()


def daily_window(g: pd.DataFrame, days: int, col: str = "discharge"):
    """Last `days` calendar days of a station, gaps kept as None (never interpolated)."""
    s = g.set_index("date")[col].sort_index()
    idx = pd.date_range(s.index.max() - pd.Timedelta(days=days - 1), s.index.max(), freq="D")
    s = s.reindex(idx)
    return [d.strftime("%Y-%m-%d") for d in idx], numlist(s.values)


def align_forecasts(fc: pd.DataFrame, obs: pd.Series, horizon: int, days: int = 365):
    """fc follows the frozen forecast contract (station_id, issue_date, horizon, q_pred[, q_low, q_high]).
    A forecast issued on D with horizon h is plotted at D+h and compared with the observation of D+h."""
    f = fc[fc["horizon"] == horizon].copy()
    if f.empty:
        return None
    f["target"] = pd.to_datetime(f["issue_date"]) + pd.Timedelta(days=horizon)
    f = f.drop_duplicates("target", keep="last").set_index("target").sort_index().tail(days)
    for c in ("q_low", "q_high"):
        if c not in f:
            f[c] = np.nan
    return {
        "horizon": horizon,
        "dates": [d.strftime("%Y-%m-%d") for d in f.index],
        "q_pred": numlist(f["q_pred"].values), "q_low": numlist(f["q_low"].values),
        "q_high": numlist(f["q_high"].values),
        "actual": numlist(obs.reindex(f.index).values),
        "last_issue_date": pd.Timestamp(f["issue_date"].iloc[-1]).strftime("%Y-%m-%d"),
        "last_target_date": f.index[-1].strftime("%Y-%m-%d"),
        "last_q_pred": num(f["q_pred"].iloc[-1]),
        "split": str(f["split"].iloc[-1]) if "split" in f else None,
    }


def weather_summary(w: pd.DataFrame) -> dict:
    """w: one station's weather_daily rows."""
    w = w.sort_values("date")
    rec = w.tail(30)
    soil = w.tail(90)
    wy = w.assign(year=w["date"].dt.year).groupby("year").agg(rain=("rain_mm", "sum"), n=("rain_mm", "size"))
    wy = wy[wy.n >= 360]                                  # complete years only, so totals are comparable
    return {
        "last_date": w["date"].max().strftime("%Y-%m-%d"),
        "rain_30d_total_mm": num(rec["rain_mm"].sum(), 1),
        "recent": {"dates": [d.strftime("%Y-%m-%d") for d in rec["date"]], "rain_mm": numlist(rec["rain_mm"], 2)},
        "soil": {"dates": [d.strftime("%Y-%m-%d") for d in soil["date"]],
                 "soil1": numlist(soil["soil1"], 4), "soil2": [None] * len(soil)},
        "annual": {"years": [int(y) for y in wy.index], "rain_mm": numlist(wy["rain"], 0)},
    }
