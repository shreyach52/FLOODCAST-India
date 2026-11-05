"""Sequence data for the LSTM (numpy/pandas only, unit-tested; torch lives in scripts/07_lstm.py).

One row per (station, day t). A window = the previous L days ending at t, so every input is dated <= t.
Dynamic channels per day:  q_anom   log1p(discharge) - log1p(station TRAIN median)   (standardised on train rows)
                           q_mask   1 if discharge was observed that day, else 0 (missing discharge is filled with 0)
                           rain     log1p(rain_mm)   [weather mode]   (standardised on train rows)
                           soil     soil moisture layer 1 [weather mode] (standardised on train rows)
                           doy_sin, doy_cos
Static per station: log1p(train median), lat, lon (standardised over SEEN stations).
Targets: residual r_h = log1p(y_{t+h}) - log1p(q_t) for h in (1, 3, 7): the same "correction to persistence" as the baseline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .evalcommon import H, metrics, per_station


@dataclass
class SeqData:
    A: np.ndarray            # (N, F) float32 dynamic features, station blocks stored back to back
    static: np.ndarray       # (S, 3) float32
    st_idx: np.ndarray       # (N,) station index into `names`
    names: list
    date: np.ndarray         # (N,) datetime64[ns]
    q0: np.ndarray           # (N,) raw discharge at t
    y: np.ndarray            # (N, 3) raw discharge at t+1, t+3, t+7 (NaN if missing)
    valid: np.ndarray        # (N,) enough history and q0 observed
    thr: np.ndarray          # (N,) station high-flow threshold (train p95)
    feat_names: list


def build_arrays(df, w, stations, tr_end, L=60, use_weather=True, min_train_obs=30, ref_end=None) -> SeqData:
    """ref_end: optional {station: last date used for its median/threshold} (late-start stations use val end)."""
    tr_end = pd.Timestamp(tr_end)
    names, blocks, stat, sts, dates, q0s, ys, valids, thrs = [], [], [], [], [], [], [], [], []
    for sid in sorted(stations):
        g = df[df.station_id == sid]
        if g.empty:
            continue
        s = g.set_index("date")["discharge"].sort_index()
        idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
        q = s.reindex(idx)
        trq = q[q.index <= pd.Timestamp((ref_end or {}).get(sid, tr_end))].dropna()
        if len(trq) < min_train_obs:
            continue
        med, thr = float(trq.median()), float(trq.quantile(0.95))
        cols = {"q_anom": (np.log1p(q) - np.log1p(med)).values, "q_mask": q.notna().astype(float).values}
        if use_weather:
            ws = w[w.station_id == sid].drop_duplicates("date").set_index("date").reindex(idx)
            cols["rain"] = np.log1p(ws["rain_mm"].clip(lower=0)).values
            cols["soil"] = ws["soil1"].values
        doy = idx.dayofyear.values
        cols["doy_sin"], cols["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
        feat_names = list(cols)
        blocks.append(np.column_stack([cols[k] for k in feat_names]))
        lat = float(g["lat"].iloc[0]) if "lat" in g else 0.0
        lon = float(g["lon"].iloc[0]) if "lon" in g else 0.0
        stat.append([np.log1p(med), lat, lon])
        k = len(names)
        names.append(sid)
        n = len(idx)
        sts.append(np.full(n, k)); dates.append(idx.values); q0s.append(q.values)
        ys.append(np.column_stack([q.shift(-h).values for h in H]))
        valids.append((np.arange(n) >= L - 1) & q.notna().values)
        thrs.append(np.full(n, thr))
    A = np.vstack(blocks)
    date = np.concatenate(dates)
    train_rows = date <= np.datetime64(tr_end)
    for j, nm in enumerate(feat_names):          # standardise on TRAIN rows only
        if nm in ("q_anom", "rain", "soil"):
            col = A[:, j]
            ok = train_rows & np.isfinite(col)
            mu, sd = col[ok].mean(), col[ok].std() + 1e-6
            A[:, j] = (col - mu) / sd
    A = np.nan_to_num(A, nan=0.0).astype(np.float32)
    static = np.array(stat, dtype=float)
    return SeqData(A=A, static=static, st_idx=np.concatenate(sts), names=names, date=date,
                   q0=np.concatenate(q0s), y=np.vstack(ys), valid=np.concatenate(valids),
                   thr=np.concatenate(thrs), feat_names=feat_names)


def standardise_static(static, st_names, held):
    seen = np.array([n not in held for n in st_names])
    mu, sd = static[seen].mean(0), static[seen].std(0) + 1e-6
    return ((static - mu) / sd).astype(np.float32)


def residual_targets(d: SeqData) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.log1p(d.y) - np.log1p(d.q0)[:, None]
    r[~d.valid] = np.nan
    return r.astype(np.float32)


def gather(A, g, L):
    """Windows for global row indices g -> (B, L, F). Rows must have local position >= L-1 (valid=True)."""
    return A[g[:, None] - (L - 1) + np.arange(L)[None, :]]


def select(d: SeqData, held, tr_end, va_end, te_start):
    """Index arrays + per-horizon target masks with the SAME rules as the baseline."""
    tr_end, va_end, te_start = (np.datetime64(pd.Timestamp(x)) for x in (tr_end, va_end, te_start))
    seen = ~np.isin(np.array(d.names)[d.st_idx], list(held))
    have = ~np.isnan(d.y)
    tdate = d.date[:, None] + np.array(H, dtype="timedelta64[D]")[None, :]
    base = d.valid[:, None] & have
    m_tr = base & seen[:, None] & (tdate <= tr_end)
    m_va = base & seen[:, None] & (d.date[:, None] > tr_end) & (tdate <= va_end)
    m_te_seen = base & seen[:, None] & (d.date[:, None] >= te_start)
    m_te_unseen = base & (~seen)[:, None] & (d.date[:, None] >= te_start)
    return {"train": m_tr, "val": m_va, "A_seen_stations": m_te_seen, "B_unseen_stations": m_te_unseen}


def evaluate(d: SeqData, masks, resid_pred, model_name="lstm"):
    """resid_pred: (N, 3) predicted residuals for ALL rows (NaN-free where a mask needs it).
    Returns (metrics DataFrame, long predictions DataFrame)."""
    rows, preds = [], []
    sid = np.array(d.names)[d.st_idx]
    for split in ("A_seen_stations", "B_unseen_stations"):
        m = masks[split]
        for i, h in enumerate(H):
            k = np.nonzero(m[:, i])[0]
            if len(k) == 0:
                continue
            te = pd.DataFrame({"station_id": sid[k], "date": d.date[k], "q0": d.q0[k], "y": d.y[k, i],
                               "thr": d.thr[k]})
            te["p_persistence"] = te.q0
            te[f"p_{model_name}"] = np.clip(np.expm1(np.log1p(te.q0.values) + resid_pred[k, i]), 0, None)
            for mname in ("persistence", model_name):
                col = f"p_{mname}"
                r = {"horizon": h, "split": split, "model": mname, **metrics(te.y, te[col], te.thr, te.q0)}
                r["median_station_nse"], r["median_station_rmse"] = per_station(te, "y", col)
                rows.append(r)
            preds.append(te.assign(horizon=h, split=split)[["station_id", "date", "horizon", "split", "y", "q0",
                                                            f"p_{model_name}"]].rename(columns={f"p_{model_name}": "p"}))
    return pd.DataFrame(rows), (pd.concat(preds, ignore_index=True) if preds else pd.DataFrame())
