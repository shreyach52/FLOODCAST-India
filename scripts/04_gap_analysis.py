"""STEP 1d: per-station gap/seasonality analysis -> which stations can support lag, sequence and 1/3/7-day models.

  python scripts/04_gap_analysis.py [--lookback 14] [--min-windows 500]
Reads data/interim/discharge_daily.csv.gz. Writes data/interim/station_windows.csv
and reports/gap_report.md.
"""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from floodcast import config

HORIZONS = (1, 3, 7)


def analyse(g: pd.DataFrame, lookback: int) -> dict:
    s = g.set_index("date")["discharge"].sort_index()
    full = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
    ok = full.notna().to_numpy()
    n = len(ok)
    # longest unbroken run of daily observations
    best = cur = 0
    for v in ok:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    # gaps longer than 30 days
    gaps = (~pd.Series(ok)).astype(int)
    grp = (gaps.diff().fillna(0) != 0).cumsum()
    runs = gaps.groupby(grp).agg(["first", "size"])
    long_gaps = int(((runs["first"] == 1) & (runs["size"] > 30)).sum())
    # complete windows: `lookback` consecutive days ending at t AND targets at t+1, t+3, t+7 all observed
    hist = pd.Series(ok).astype(int).rolling(lookback).sum().eq(lookback).to_numpy()
    tgt = np.ones(n, dtype=bool)
    for h in HORIZONS:
        t = np.zeros(n, dtype=bool)
        t[:-h] = ok[h:]
        tgt &= t
    windows = int((hist & tgt).sum())
    months = s.index.month
    return {
        "n_obs": len(s), "span_years": round(n / 365.25, 2),
        "longest_run_days": best, "gaps_over_30d": long_gaps,
        "windows_1_3_7": windows,
        "months_present": int(pd.Series(months).nunique()),
        "monsoon_share": round(float(((months >= 6) & (months <= 10)).mean()), 2),
        "q_max": round(float(s.max()), 2), "q_median": round(float(s.median()), 2),
        "cv": round(float(s.std() / s.mean()), 2) if s.mean() > 0 else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=str(config.INTERIM))
    ap.add_argument("--lookback", type=int, default=14)
    ap.add_argument("--min-windows", type=int, default=500)
    a = ap.parse_args()
    df = pd.read_csv(Path(a.in_dir) / "discharge_daily.csv.gz", parse_dates=["date"])
    rows = [{"station_id": k, **analyse(g, a.lookback)} for k, g in df.groupby("station_id")]
    out = pd.DataFrame(rows).set_index("station_id")
    out["sequence_ready"] = out["windows_1_3_7"] >= a.min_windows
    out["seasonal_only"] = out["months_present"] < 12
    out.to_csv(Path(a.in_dir) / "station_windows.csv")

    ready = out[out.sequence_ready]
    md = [f"# Gap report (lookback={a.lookback}d, horizons 1/3/7, min windows={a.min_windows})\n",
          f"- Stations: **{len(out)}**; sequence-ready: **{len(ready)}**; "
          f"seasonal-only (<12 months present): **{int(out.seasonal_only.sum())}**",
          f"- Total usable windows in ready stations: **{int(ready.windows_1_3_7.sum()):,}**\n",
          "## Not sequence-ready (bottom 25 by windows)\n", "```",
          out[~out.sequence_ready].sort_values("windows_1_3_7").head(25).to_string(), "```\n",
          "## Sequence-ready summary\n", "```", ready.describe().round(2).to_string(), "```"]
    p = config.REPORTS / "gap_report.md" if hasattr(config, "REPORTS") else Path("reports/gap_report.md")
    p.parent.mkdir(exist_ok=True)
    p.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:3]))
    print(f"Wrote {p} and station_windows.csv")


if __name__ == "__main__":
    main()