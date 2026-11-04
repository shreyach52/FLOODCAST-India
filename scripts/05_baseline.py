"""STEP 2: discharge-only baseline for 1/3/7-day-ahead forecasts (LightGBM vs persistence vs climatology).

  python scripts/05_baseline.py
Two evaluations, both on FUTURE dates (test = after --test-start):
  A) seen stations   : same stations as training, later dates
  B) unseen stations : a random ~15% of stations never used in training
Leakage rules: features use only values at or before day t; a training row's TARGET date must be
<= train end (so no future value leaks across the split boundary).
Outputs reports/baseline_report.md and data/interim/baseline_metrics.csv
"""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from floodcast import config

H = (1, 3, 7)
LAGS = (1, 2, 3, 5, 7, 14)


def make_features(g: pd.DataFrame) -> pd.DataFrame:
    s = g.set_index("date")["discharge"].sort_index()
    q = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
    f = pd.DataFrame({"q0": q})
    for k in LAGS:
        f[f"lag{k}"] = q.shift(k)
    for w in (7, 14, 30):
        f[f"mean{w}"] = q.rolling(w, min_periods=w).mean()
    f["max7"] = q.rolling(7, min_periods=7).max()
    f["diff1"] = q - q.shift(1)
    f["diff3"] = q - q.shift(3)
    f["ratio7"] = q / (f["mean7"] + 1e-6)
    doy = f.index.dayofyear
    f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    for h in H:
        f[f"y{h}"] = q.shift(-h)
    f.index.name = "date"
    return f.reset_index()


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
            on = hi & (q0 < thr)                      # below threshold today, above it at the horizon
            fa = pred_hi & (q0 < thr)                 # forecast says "will cross" while today is below
            out["onset_n"] = int(on.sum())
            out["onset_recall"] = float((fa & on).sum() / on.sum()) if on.any() else np.nan
            out["onset_alarms"] = int(fa.sum())
            out["onset_precision"] = float((fa & on).sum() / fa.sum()) if fa.any() else np.nan
    return out


def median_station_nse(df, ycol, pcol):
    r = []
    for _, g in df.groupby("station_id"):
        d = ((g[ycol] - g[ycol].mean()) ** 2).sum()
        if len(g) >= 30 and d > 0:
            r.append(1 - ((g[ycol] - g[pcol]) ** 2).sum() / d)
    return float(np.median(r)) if r else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=str(config.INTERIM))
    ap.add_argument("--train-end", default="2018-12-31")
    ap.add_argument("--val-end", default="2021-12-31")
    ap.add_argument("--test-start", default="2022-01-01")
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target", choices=["residual", "log"], default="residual",
                    help="residual: learn log1p(y)-log1p(q0), i.e. a correction to persistence (default)")
    a = ap.parse_args()
    ind = Path(a.in_dir)
    df = pd.read_csv(ind / "discharge_daily.csv.gz", parse_dates=["date"])
    win = pd.read_csv(ind / "station_windows.csv")
    ready = set(win.loc[win.sequence_ready, "station_id"])
    df = df[df.station_id.isin(ready)]
    static = df.groupby("station_id")[[c for c in ("lat", "lon") if c in df.columns]].first()

    parts = []
    for sid, g in df.groupby("station_id"):
        f = make_features(g)
        f.insert(0, "station_id", sid)
        parts.append(f)
    X = pd.concat(parts, ignore_index=True).join(static, on="station_id")
    X = X[X.q0.notna()]
    tr_med = X[X.date <= pd.Timestamp(a.train_end)].groupby("station_id").q0.median()
    X["q_rel_med"] = X.q0 / (X.station_id.map(tr_med) + 1e-6)
    X["mean7_rel_med"] = X.mean7 / (X.station_id.map(tr_med) + 1e-6)
    X["log_station_med"] = np.log1p(X.station_id.map(tr_med))
    feats = [c for c in X.columns if c not in ("station_id", "date", "thr") and not c.startswith("y")]

    rng = np.random.default_rng(a.seed)
    sids = np.array(sorted(ready))
    held = set(rng.choice(sids, size=max(1, int(len(sids) * a.holdout_frac)), replace=False))
    tr_end, va_end, te_start = map(pd.Timestamp, (a.train_end, a.val_end, a.test_start))

    # per-station high-flow threshold = 95th percentile of TRAIN-period discharge of seen stations
    # (for held-out stations it is computed from their data before the train end: history is legitimately known)
    thr = X[X.date <= tr_end].groupby("station_id").q0.quantile(0.95)
    X["thr"] = X.station_id.map(thr)
    # day-of-year climatology (train period only) as a second baseline
    clim = X[X.date <= tr_end].assign(doy=lambda d: d.date.dt.dayofyear).groupby(["station_id", "doy"]).q0.median()

    rows, lines = [], []
    for h in H:
        y = f"y{h}"
        d = X[X[y].notna()].copy()
        d["tdate"] = d.date + pd.Timedelta(days=h)
        seen = ~d.station_id.isin(held)
        tr = d[seen & (d.tdate <= tr_end)]
        va = d[seen & (d.date > tr_end) & (d.tdate <= va_end)]
        te_seen = d[seen & (d.date >= te_start)]
        te_unseen = d[~seen & (d.date >= te_start)]
        p = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50,
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=a.seed)
        # model in log space: discharge is heavy-tailed and stations differ by orders of magnitude
        base = (lambda d: np.log1p(d.q0)) if a.target == "residual" else (lambda d: 0.0)
        m = lgb.train(p, lgb.Dataset(tr[feats], np.log1p(tr[y]) - base(tr)), 2000,
                      valid_sets=[lgb.Dataset(va[feats], np.log1p(va[y]) - base(va))],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        for name, te in (("A_seen_stations", te_seen), ("B_unseen_stations", te_unseen)):
            if te.empty:
                continue
            te = te.copy()
            te["p_lgb"] = np.clip(np.expm1(m.predict(te[feats], num_iteration=m.best_iteration) + np.asarray(base(te))), 0, None)
            te["p_persist"] = te.q0
            doy = te.date.dt.dayofyear + h
            doy = doy.where(doy <= 366, doy - 366)
            te["p_clim"] = [clim.get((s, int(k)), np.nan) for s, k in zip(te.station_id, doy)]
            te["p_clim"] = te.p_clim.fillna(te.q0)
            for mname, col in (("persistence", "p_persist"), ("climatology", "p_clim"), ("lightgbm", "p_lgb")):
                r = {"horizon": h, "split": name, "model": mname, **metrics(te[y], te[col], te.thr, te.q0)}
                r["median_station_nse"] = median_station_nse(te, y, col)
                rows.append(r)

    res = pd.DataFrame(rows)
    res.to_csv(ind / "baseline_metrics.csv", index=False)
    show = res[["horizon", "split", "model", "n", "rmse", "mae", "median_station_nse",
                "hi_n", "hi_rmse", "hi_recall", "hi_precision",
                "onset_n", "onset_recall", "onset_alarms", "onset_precision"]].round(3)
    md = ["# Baseline report (discharge-only)\n",
          f"Train <= {a.train_end}, val <= {a.val_end}, test >= {a.test_start}. "
          f"Held-out stations ({len(held)}): {sorted(map(str, held))}\n",
          "High flow = station's train-period 95th percentile. hi_recall = share of true high-flow days "
          "whose forecast also reached that level; hi_precision = share of such alarms that were real.\n",
          "```", show.to_string(index=False), "```"]
    (config.REPORTS / "baseline_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()