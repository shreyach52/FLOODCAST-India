"""Evaluation helpers shared by the LSTM script. Logic is copied 1:1 from scripts/05_baseline.py so that
LSTM and LightGBM are scored on the same stations, the same split dates and the same metrics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

H = (1, 3, 7)
REGULATED = r"reservoir|seasonal"


def ready_stations(in_dir, exclude=()) -> set:
    win = pd.read_csv(Path(in_dir) / "station_windows.csv")
    ready = set(win.loc[win.sequence_ready, "station_id"])
    ready = {s for s in ready if not pd.Series([s]).str.contains(REGULATED, case=False, regex=True).iloc[0]}
    return ready - set(exclude)


def choose_holdout(ready, frac=0.15, seed=42) -> set:
    rng = np.random.default_rng(seed)
    sids = np.array(sorted(ready))
    return set(rng.choice(sids, size=max(1, int(len(sids) * frac)), replace=False))


def station_groups(df, ready, tr_end, va_end, frac=0.15, seed=42, min_obs=30):
    """Split stations into trained-on and unseen groups.
    hist  : >= min_obs observations on/before train end (can be trained on).
    held  : random `frac` of hist, never trained on (station-held-out test).
    late  : no usable history before train end (record starts later); never trained on, treated as unseen. Their
            reference statistics (median, p95 threshold) come from data up to val end, i.e. known before the test starts.
    dropped: fewer than min_obs observations even up to val end.
    ref_end: per-station last date used for reference statistics."""
    tr_end, va_end = pd.Timestamp(tr_end), pd.Timestamp(va_end)
    d = df[df.station_id.isin(ready)]
    n_tr = d[d.date <= tr_end].groupby("station_id").size()
    n_va = d[d.date <= va_end].groupby("station_id").size()
    hist = {s for s in ready if n_tr.get(s, 0) >= min_obs}
    held = choose_holdout(hist, frac, seed)
    late = {s for s in ready - hist if n_va.get(s, 0) >= min_obs}
    return {"hist": hist, "held": held, "late": late, "dropped": set(ready) - hist - late,
            "unseen": held | late, "ref_end": {s: (tr_end if s in hist else va_end) for s in hist | late}}


def metrics(y, p, thr=None, q0=None):
    y, p = np.asarray(y, float), np.asarray(p, float)
    out = {"n": len(y), "rmse": float(np.sqrt(np.mean((y - p) ** 2))), "mae": float(np.mean(np.abs(y - p)))}
    if thr is not None:
        hi = y >= thr
        out["hi_n"] = int(hi.sum())
        out["hi_rmse"] = float(np.sqrt(np.mean((y[hi] - p[hi]) ** 2))) if hi.any() else np.nan
        pred_hi = p >= thr
        out["hi_recall"] = float((pred_hi & hi).sum() / hi.sum()) if hi.any() else np.nan
        out["hi_precision"] = float((pred_hi & hi).sum() / pred_hi.sum()) if pred_hi.any() else np.nan
        if q0 is not None:
            q0 = np.asarray(q0, float)
            on = hi & (q0 < thr)
            fa = pred_hi & (q0 < thr)
            out["onset_n"] = int(on.sum())
            out["onset_recall"] = float((fa & on).sum() / on.sum()) if on.any() else np.nan
            out["onset_alarms"] = int(fa.sum())
            out["onset_precision"] = float((fa & on).sum() / fa.sum()) if fa.any() else np.nan
    return out


def per_station(df, ycol, pcol):
    nse, rm = [], []
    for _, g in df.groupby("station_id"):
        if len(g) < 30:
            continue
        d = ((g[ycol] - g[ycol].mean()) ** 2).sum()
        if d > 0:
            nse.append(1 - ((g[ycol] - g[pcol]) ** 2).sum() / d)
        rm.append(np.sqrt(((g[ycol] - g[pcol]) ** 2).mean()))
    return (float(np.median(nse)) if nse else np.nan), (float(np.median(rm)) if rm else np.nan)


ORDER = ["persistence", "lightgbm", "lightgbm_wx", "lstm"]
COLS = ["horizon", "split", "model", "n", "rmse", "median_station_rmse", "median_station_nse", "hi_n", "hi_rmse",
        "hi_recall", "hi_precision", "onset_n", "onset_recall", "onset_alarms", "onset_precision"]


def combine_with_baseline(res: pd.DataFrame, baseline_csv) -> pd.DataFrame:
    """LSTM metric rows + the LightGBM rows of baseline_metrics.csv (if present), sorted for display."""
    allr = res.copy()
    p = Path(baseline_csv)
    if p.exists():
        b = pd.read_csv(p)
        allr = pd.concat([res, b[b.model.isin(["lightgbm", "lightgbm_wx"])]], ignore_index=True)
    for c in COLS:
        if c not in allr:
            allr[c] = np.nan
    allr["model"] = allr["model"].astype(str)
    allr["_o"] = allr["model"].map({m: i for i, m in enumerate(ORDER)})
    return allr.sort_values(["horizon", "split", "_o"]).drop(columns="_o")[COLS].reset_index(drop=True)


def delta_table(allr: pd.DataFrame):
    """LSTM minus the best available LightGBM variant (lightgbm_wx if it has numbers, else lightgbm). None if no baseline."""
    models = set(allr.model)
    ref = "lightgbm_wx" if "lightgbm_wx" in models else ("lightgbm" if "lightgbm" in models else None)
    if ref is None or "lstm" not in models:
        return None, None
    k = allr[allr.model.isin(["lstm", ref])].pivot(index=["horizon", "split"], columns="model",
                                                  values=["median_station_nse", "onset_recall", "onset_precision"])
    out = pd.DataFrame({f"nse_lstm_minus_{ref}": k[("median_station_nse", "lstm")] - k[("median_station_nse", ref)],
                        f"onset_recall_lstm_minus_{ref}": k[("onset_recall", "lstm")] - k[("onset_recall", ref)],
                        f"onset_prec_lstm_minus_{ref}": k[("onset_precision", "lstm")] - k[("onset_precision", ref)]})
    return out.round(3), ref
