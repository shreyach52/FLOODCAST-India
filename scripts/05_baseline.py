"""STEP 2/3: 1/3/7-day-ahead baseline. LightGBM discharge-only vs LightGBM + weather vs persistence vs climatology.

  python scripts/05_baseline.py                 # uses data/interim/weather_daily.csv.gz if it exists
  python scripts/05_baseline.py --no-weather    # discharge-only, as before
  python scripts/05_baseline.py --exclude Yadgir Haralahalli Huvinhedigi   # optional: drop stations from everything
Two evaluations, both on FUTURE dates (test = after --test-start):
  A) seen stations   : same stations as training, later dates
  B) unseen stations : a random ~15% of stations never used in training
Leakage rules: features use only values at or before day t; a training row's TARGET date must be <= train end.
Weather features use only weather dated <= t and NO forecast rain (see floodcast/features_wx.py).
Reservoir / seasonal (regulated) stations are always excluded: their flow is released, not rain-driven.
Outputs reports/baseline_report.md and data/interim/baseline_metrics.csv
"""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from floodcast import config
from floodcast.evalcommon import station_groups
from floodcast.features_wx import WX_FEATURES, weather_features

H = (1, 3, 7)
LAGS = (1, 2, 3, 5, 7, 14)
REGULATED = r"reservoir|seasonal"


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


def per_station(df, ycol, pcol):
    """(median station NSE, median station RMSE). Both robust to a few flashy / dubious stations."""
    nse, rm = [], []
    for _, g in df.groupby("station_id"):
        if len(g) < 30:
            continue
        d = ((g[ycol] - g[ycol].mean()) ** 2).sum()
        if d > 0:
            nse.append(1 - ((g[ycol] - g[pcol]) ** 2).sum() / d)
        rm.append(np.sqrt(((g[ycol] - g[pcol]) ** 2).mean()))
    return (float(np.median(nse)) if nse else np.nan), (float(np.median(rm)) if rm else np.nan)


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
    ap.add_argument("--weather", default=None, help="path to weather_daily.csv.gz (default: <in-dir>/weather_daily.csv.gz)")
    ap.add_argument("--no-weather", action="store_true")
    ap.add_argument("--exclude", nargs="*", default=[], help="station_id values to drop entirely")
    a = ap.parse_args()
    ind = Path(a.in_dir)
    df = pd.read_csv(ind / "discharge_daily.csv.gz", parse_dates=["date"])
    win = pd.read_csv(ind / "station_windows.csv")
    ready = set(win.loc[win.sequence_ready, "station_id"])
    ready = {s for s in ready if not pd.Series([s]).str.contains(REGULATED, case=False, regex=True).iloc[0]}
    ready -= set(a.exclude)
    df = df[df.station_id.isin(ready)]
    tr_end, va_end, te_start = map(pd.Timestamp, (a.train_end, a.val_end, a.test_start))
    grp = station_groups(df, ready, tr_end, va_end, a.holdout_frac, a.seed)
    ready = grp["hist"] | grp["late"]
    df = df[df.station_id.isin(ready)]
    print(f"stations: {len(grp['hist']) - len(grp['held'])} trained-on, {len(grp['held'])} random held-out, "
          f"{len(grp['late'])} late-start (no training-period data), {len(grp['dropped'])} dropped (too little data)")
    static = df.groupby("station_id")[[c for c in ("lat", "lon") if c in df.columns]].first()

    parts = []
    for sid, g in df.groupby("station_id"):
        f = make_features(g)
        f.insert(0, "station_id", sid)
        parts.append(f)
    X = pd.concat(parts, ignore_index=True).join(static, on="station_id")
    X = X[X.q0.notna()]
    # reference statistics: train period for stations with history; up to val end for late-start stations
    ref = X[X.date <= X.station_id.map(grp["ref_end"])]
    tr_med = ref.groupby("station_id").q0.median()
    X["q_rel_med"] = X.q0 / (X.station_id.map(tr_med) + 1e-6)
    X["mean7_rel_med"] = X.mean7 / (X.station_id.map(tr_med) + 1e-6)
    X["log_station_med"] = np.log1p(X.station_id.map(tr_med))
    feats = [c for c in X.columns if c not in ("station_id", "date", "thr") and not c.startswith("y")]

    # ---- weather join (optional) ----
    wpath = Path(a.weather) if a.weather else ind / "weather_daily.csv.gz"
    feats_wx = None
    if not a.no_weather and wpath.exists():
        w = pd.read_csv(wpath, parse_dates=["date"])
        wf = weather_features(w[w.station_id.isin(ready)], tr_end)
        X = X.merge(wf, on=["station_id", "date"], how="left")
        feats_wx = feats + list(WX_FEATURES)
        cover = X[list(WX_FEATURES)].notna().all(axis=1).mean()
        print(f"weather joined from {wpath.name}; rows with complete weather features: {cover:.1%}")
    elif not a.no_weather:
        print(f"NOTE: {wpath} not found -> running discharge-only. Run 06a/06b first.")

    held = grp["unseen"]   # random held-out stations + late-start stations: never trained on

    thr = ref.groupby("station_id").q0.quantile(0.95)
    X["thr"] = X.station_id.map(thr)
    clim = ref.assign(doy=lambda d: d.date.dt.dayofyear).groupby(["station_id", "doy"]).q0.median()

    models = [("lightgbm", feats)] + ([("lightgbm_wx", feats_wx)] if feats_wx else [])
    rows, imp_lines = [], []
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
        base = (lambda d: np.log1p(d.q0)) if a.target == "residual" else (lambda d: 0.0)
        fitted = {}
        for mname, fl in models:
            m = lgb.train(p, lgb.Dataset(tr[fl], np.log1p(tr[y]) - base(tr)), 2000,
                          valid_sets=[lgb.Dataset(va[fl], np.log1p(va[y]) - base(va))],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            fitted[mname] = (m, fl)
        if "lightgbm_wx" in fitted:
            m, fl = fitted["lightgbm_wx"]
            gain = pd.Series(m.feature_importance("gain"), index=fl).sort_values(ascending=False)
            imp_lines.append(f"h={h}d top features (gain share): " +
                             ", ".join(f"{k} {v / gain.sum():.0%}" for k, v in gain.head(8).items()))
        for name, te in (("A_seen_stations", te_seen), ("B_unseen_stations", te_unseen)):
            if te.empty:
                continue
            te = te.copy()
            for mname, (m, fl) in fitted.items():
                te[f"p_{mname}"] = np.clip(np.expm1(m.predict(te[fl], num_iteration=m.best_iteration) + np.asarray(base(te))), 0, None)
            te["p_persist"] = te.q0
            doy = te.date.dt.dayofyear + h
            doy = doy.where(doy <= 366, doy - 366)
            te["p_clim"] = [clim.get((s, int(k)), np.nan) for s, k in zip(te.station_id, doy)]
            te["p_clim"] = te.p_clim.fillna(te.q0)
            cols = [("persistence", "p_persist"), ("climatology", "p_clim")] + [(m_, f"p_{m_}") for m_, _ in models]
            for mname, col in cols:
                r = {"horizon": h, "split": name, "model": mname, **metrics(te[y], te[col], te.thr, te.q0)}
                r["median_station_nse"], r["median_station_rmse"] = per_station(te, y, col)
                rows.append(r)

    res = pd.DataFrame(rows)
    res.to_csv(ind / "baseline_metrics.csv", index=False)
    show = res[["horizon", "split", "model", "n", "rmse", "median_station_rmse", "median_station_nse",
                "hi_n", "hi_rmse", "hi_recall", "hi_precision",
                "onset_n", "onset_recall", "onset_alarms", "onset_precision"]].round(3)
    md = [f"# Baseline report ({'discharge + ERA5-Land weather' if feats_wx else 'discharge-only'})\n",
          f"Train <= {a.train_end}, val <= {a.val_end}, test >= {a.test_start}. Stations used: {len(ready)}. "
          f"Unseen stations: {len(grp['held'])} random held-out {sorted(map(str, grp['held']))} + "
          f"{len(grp['late'])} late-start (record begins after train end, no training data) {sorted(map(str, grp['late']))}\n",
          "High flow = station's train-period 95th percentile. hi_recall = share of true high-flow days whose forecast also "
          "reached that level. onset_* = only days where the river is BELOW its threshold today (persistence scores 0 by "
          "construction): onset_recall = share of true crossings flagged in advance, onset_precision = share of alarms that were real. "
          "median_station_rmse / median_station_nse are robust to a few flashy stations; judge progress with them and onset_*.\n",
          "```", show.to_string(index=False), "```"]
    if feats_wx:
        k = res[res.model.isin(["lightgbm", "lightgbm_wx"])].pivot(
            index=["horizon", "split"], columns="model",
            values=["median_station_rmse", "median_station_nse", "onset_recall", "onset_precision", "onset_alarms"])
        delta = pd.DataFrame({
            "med_rmse_change": k[("median_station_rmse", "lightgbm_wx")] - k[("median_station_rmse", "lightgbm")],
            "nse_change": k[("median_station_nse", "lightgbm_wx")] - k[("median_station_nse", "lightgbm")],
            "onset_recall_wo": k[("onset_recall", "lightgbm")], "onset_recall_wx": k[("onset_recall", "lightgbm_wx")],
            "onset_prec_wo": k[("onset_precision", "lightgbm")], "onset_prec_wx": k[("onset_precision", "lightgbm_wx")],
        }).round(3)
        md += ["", "## What the weather adds (lightgbm_wx minus lightgbm; negative RMSE change = better)", "```",
               delta.to_string(), "```", "", "## Feature importance (weather model)", "```", *imp_lines, "```"]
    (config.REPORTS / "baseline_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
