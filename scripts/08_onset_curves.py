"""STEP 4b: compare models on early warning WITHOUT depending on one alarm rule.

Input: a predictions file (csv / csv.gz) with ONE ROW PER (station, date, horizon) holding, at least:
  station, horizon, split label, q0 (discharge today), y (true discharge h days ahead),
  thr (that station's high-flow threshold), and one column of predicted discharge h days ahead per model.
Example:
  python scripts/08_onset_curves.py --pred data/interim/lstm_test_predictions.csv.gz \
      --model lstm=pred_lstm --model lightgbm_wx=pred_lgb_wx
If the models live in two files, merge the second one in on (station, date, horizon):
  --pred2 other.csv.gz --model2 lightgbm_wx=pred
Column names are arguments because they differ between files; a wrong name prints the real header.
Outputs: reports/onset_curves.md and data/interim/onset_summary.csv
"""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

import pandas as pd

from floodcast.onset_eval import compare_pairs, prepare, summarize_by


def kv(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--model expects NAME=COLUMN, got {it!r}")
        k, v = it.split("=", 1)
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--model", action="append", required=True, help="NAME=COLUMN (repeatable)")
    ap.add_argument("--pred2")
    ap.add_argument("--model2", action="append")
    ap.add_argument("--station-col", default="station_id")
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--horizon-col", default="horizon")
    ap.add_argument("--split-col", default="split")
    ap.add_argument("--q0-col", default="q0")
    ap.add_argument("--y-col", default="y")
    ap.add_argument("--thr-col", default="thr")
    ap.add_argument("--pairs", nargs="*", help="A:B model pairs for the bootstrap; default = every consecutive pair")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--out-md", default="reports/onset_curves.md")
    ap.add_argument("--out-csv", default="data/interim/onset_summary.csv")
    a = ap.parse_args()

    df = pd.read_csv(a.pred)
    models = kv(a.model)
    keys = [a.station_col, a.date_col, a.horizon_col]
    if a.pred2:
        df2 = pd.read_csv(a.pred2)
        m2 = kv(a.model2)
        miss = [c for c in keys + list(m2.values()) if c not in df2.columns]
        if miss:
            raise SystemExit(f"--pred2 is missing {miss}. Its columns: {list(df2.columns)}")
        ren = {v: f"__m2_{k}" for k, v in m2.items()}          # unique names, so no clash with file 1
        sub = df2[keys + list(m2.values())].rename(columns=ren).drop_duplicates(keys)
        df = df.merge(sub, on=keys, how="inner")
        models.update({k: f"__m2_{k}" for k in m2})
        print(f"after merging the second file: {len(df):,} rows")
    for c in keys + [a.split_col, a.q0_col, a.y_col, a.thr_col] + list(models.values()):
        if c not in df.columns:
            raise SystemExit(f"column {c!r} not found. Columns in the file: {list(df.columns)}")
    df = df.rename(columns={a.station_col: "station_id", a.horizon_col: "horizon", a.split_col: "split"})

    d = prepare(df, models, q0=a.q0_col, y=a.y_col, thr=a.thr_col)
    print(f"onset-eligible rows: {len(d):,} of {len(df):,}; onsets: {int(d['_label'].sum()):,}")
    summary = summarize_by(d, by=("horizon", "split"))
    names = list(models)
    pairs = [tuple(p.split(":")) for p in a.pairs] if a.pairs else list(zip(names[:-1], names[1:]))
    pairs += [(n, "persistence") for n in names]
    comp = compare_pairs(d, pairs, n_boot=a.n_boot)

    Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_md).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(a.out_csv, index=False)
    md = ["# Threshold-free early-warning comparison", "",
          "Onset = river below its high-flow threshold today and at/above it h days later. Score = predicted "
          "discharge / threshold. `persistence` is q0/threshold (a real score here).", "",
          "- **roc_auc**: 0.5 = coin flip, 1 = perfect ranking.",
          "- **pr_auc**: compare with `prevalence` (the score of a random guess).",
          "- **tpr@F**: share of onsets caught while only F of non-onset days may raise a false alarm.",
          "- Differences are only meaningful when the 95% interval (lo, hi) excludes 0.", "",
          "## Per model", "", "```", summary.to_string(index=False), "```", "",
          "## Paired differences (A minus B, station-block bootstrap)", "", "```",
          comp.to_string(index=False), "```", ""]
    Path(a.out_md).write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[-9:]))
    print(f"\nWrote {a.out_md} and {a.out_csv}")


if __name__ == "__main__":
    main()
