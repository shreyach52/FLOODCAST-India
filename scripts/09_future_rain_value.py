"""STEP 5: how much is knowing FUTURE rain worth for early warning?   (DIAGNOSTIC - not a deployable model)

Same LightGBM, same features, same splits and station groups as the main evaluation, trained once WITHOUT
future rain ("wx") and again with future rain of forecast skill rho in --rhos (rho=1 is the oracle that knows
the rain that really fell on days t+1..t+h). The gain over "wx" is the most any rainfall forecast could add.

  python scripts/09_future_rain_value.py --quick          # ~minutes, fewer trees/stations: check it runs
  python scripts/09_future_rain_value.py                   # full run

Needs in --in-dir (default data/interim): discharge_daily.csv.gz, weather_daily.csv.gz, station_windows.csv
Uses floodcast.evalcommon (splits, groups, metrics), floodcast.features_wx and floodcast.onset_eval.
Outputs: reports/future_rain_value.md and data/interim/future_rain_value.csv
"""
import _bootstrap  # noqa: F401
import argparse
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from floodcast.evalcommon import metrics, per_station, ready_stations, station_groups
from floodcast.features_wx import WX_FEATURES, weather_features
from floodcast.futurerain import (DIS_FEATURES, H, discharge_frame, future_rain, oracle_cols,
                                  zscore_log, degrade)
from floodcast.onset_eval import compare_pairs, prepare, summarize_by

# newer LightGBM renames eval_set; older versions only know eval_set, so keep it and hide the notice
warnings.filterwarnings("ignore", message=".*eval_set.*")


def vname(rho: float) -> str:
    return "oracle" if rho >= 1.0 else f"fc_rho{rho:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/interim")
    ap.add_argument("--tr-end", default="2018-12-31")
    ap.add_argument("--va-end", default="2021-12-31")
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.5, 0.7, 0.9, 1.0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--out-md", default="reports/future_rain_value.md")
    ap.add_argument("--out-csv", default="data/interim/future_rain_value.csv")
    a = ap.parse_args()
    tr_end, va_end = pd.Timestamp(a.tr_end), pd.Timestamp(a.va_end)
    in_dir = Path(a.in_dir)

    dis = pd.read_csv(in_dir / "discharge_daily.csv.gz", parse_dates=["date"])
    wx = pd.read_csv(in_dir / "weather_daily.csv.gz", parse_dates=["date"])
    ready = ready_stations(in_dir, exclude=a.exclude)
    grp = station_groups(dis, ready, tr_end, va_end, seed=a.seed)
    use = grp["hist"] | grp["late"]
    if a.quick:
        keep = set(sorted(grp["hist"] - grp["held"])[:20]) | set(sorted(grp["held"])[:5]) | set(sorted(grp["late"])[:5])
        use = keep
    seen = (grp["hist"] - grp["held"]) & use
    unseen = (grp["held"] | grp["late"]) & use
    print(f"stations: seen {len(seen)}, unseen {len(unseen)} (held-out {len(grp['held'] & use)}, late-start {len(grp['late'] & use)})")

    # ---- feature table
    dis = dis[dis.station_id.isin(use)]
    X = pd.concat([discharge_frame(g.assign(station_id=sid)) for sid, g in dis.groupby("station_id")], ignore_index=True)
    X = X.merge(weather_features(wx[wx.station_id.isin(use)], tr_end), on=["station_id", "date"], how="left")
    fut = future_rain(wx[wx.station_id.isin(use)])
    X = X.merge(fut, on=["station_id", "date"], how="left")
    ref_end = {s: grp["ref_end"][s] for s in use}
    thr = (dis.assign(_r=dis.date <= dis.station_id.map(ref_end)).query("_r")
           .groupby("station_id").discharge.quantile(0.95))
    X["thr"] = X.station_id.map(thr)
    X = X.dropna(subset=["q0", "thr"]).reset_index(drop=True)

    # ---- oracle features -> standardised -> degraded to each skill level (SAME noise for every rho)
    allo = sorted({c for h in H for c in oracle_cols(h)})
    Z = zscore_log(X, allo, ref_end)
    rng = np.random.default_rng(a.seed + 1)
    EPS = {c: rng.standard_normal(len(X)) for c in allo}
    variants = ["wx"] + [vname(r) for r in a.rhos]

    params = dict(n_estimators=150 if a.quick else 600, learning_rate=0.05, num_leaves=63, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, min_child_samples=50, random_state=a.seed,
                  n_jobs=-1, verbose=-1)
    base_feats = DIS_FEATURES + list(WX_FEATURES)
    lq0 = np.log1p(X.q0)
    test_start = va_end + pd.Timedelta(days=1)
    frames = []
    for h in H:
        y = X[f"y{h}"]
        tgt_date = X.date + pd.Timedelta(days=h)
        res = np.log1p(y) - lq0
        m_train = X.station_id.isin(seen) & y.notna() & (tgt_date <= tr_end)
        m_val = X.station_id.isin(seen) & y.notna() & (X.date > tr_end) & (tgt_date <= va_end)
        m_test = (X.date >= test_start) & y.notna()
        out = X.loc[m_test, ["station_id", "date", "q0", "thr"]].copy()
        out["horizon"], out["y"] = h, y[m_test]
        out["split"] = np.where(out.station_id.isin(seen), "A", "B")
        for rho, name in zip([None] + list(a.rhos), variants):
            F = X[base_feats].copy()
            if rho is not None:
                for c in oracle_cols(h):
                    F[c] = degrade(Z[c].to_numpy(), rho, EPS[c])
            m = lgb.LGBMRegressor(**params)
            m.fit(F[m_train], res[m_train], eval_set=[(F[m_val], res[m_val])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            pred = np.expm1(lq0[m_test] + m.predict(F[m_test])).clip(lower=0)
            out[f"p_{name}"] = pred.to_numpy()
            print(f"h={h} {name:<10} trees={m.best_iteration_} train={int(m_train.sum()):,} test={int(m_test.sum()):,}", flush=True)
        frames.append(out)
    P = pd.concat(frames, ignore_index=True)

    # ---- threshold-free comparison (+ paired differences against "wx")
    models = {n: f"p_{n}" for n in variants}
    D = prepare(P, models)
    summary = summarize_by(D, by=("horizon", "split"))
    comp = compare_pairs(D, [(n, "wx") for n in variants[1:]], metrics=("roc_auc", "tpr@0.02"),
                         n_boot=20 if a.quick else a.n_boot)

    # ---- classic metrics (same definitions as the main evaluation)
    rows = []
    for (h, sp), g in P.groupby(["horizon", "split"]):
        for n in ["persistence"] + variants:
            p = g["q0"] if n == "persistence" else g[f"p_{n}"]
            mm = metrics(g.y, p, g.thr, g.q0)
            nse, rm = per_station(g.assign(_p=p), "y", "_p")
            rows.append({"horizon": h, "split": sp, "model": n, "median_station_nse": round(nse, 3),
                         "onset_recall": round(mm.get("onset_recall", np.nan), 3),
                         "onset_precision": round(mm.get("onset_precision", np.nan), 3)})
    classic = pd.DataFrame(rows)

    Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_md).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(a.out_csv, index=False)
    md = ["# Value of future rain (diagnostic, NOT a deployable model)", "",
          f"Train <= {a.tr_end}, val <= {a.va_end}, test >= {test_start.date()}. Quick mode: {a.quick}.",
          f"Stations: {len(seen)} seen (split A), {len(unseen)} unseen (split B).", "",
          "`wx` = discharge + past weather only. `fc_rhoX` = also a rain forecast for days t+1..t+h whose correlation "
          "with the rain that really fell is X. `oracle` = perfect knowledge (X=1). All variants use the same model, "
          "features and splits; only the future-rain features differ. Forecast errors here are random noise of the "
          "stated correlation: real forecasts also have biases and miss extremes, so treat the curve as optimistic.", "",
          "## Threshold-free early warning (onset = below threshold today, at/above it h days later)", "```",
          summary.to_string(index=False), "```", "",
          "## Gain over `wx`, with station-block 95% interval (A minus B)", "```",
          comp.to_string(index=False), "```", "",
          "## Classic metrics at the fixed alarm rule (for continuity with the main tables)", "```",
          classic.to_string(index=False), "```", ""]
    Path(a.out_md).write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {a.out_md} and {a.out_csv}")


if __name__ == "__main__":
    main()
