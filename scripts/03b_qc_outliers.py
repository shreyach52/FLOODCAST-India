"""STEP 2b: flag and remove impossible discharge spikes BEFORE modelling.

  python scripts/03b_qc_outliers.py            # writes cleaned table, keeps a raw backup
Rules:
  DROP   : value > 20 x the station's 99.9th percentile (absurd versus its own history, e.g. a typo).
  REVIEW : value > 30 x the median of the +-3 neighbouring days AND > 500. These are NOT dropped:
           real floods (e.g. Aug 2018 / Aug 2019) can look like this. A human decides.
All flags go to reports/qc_flags.csv (column `action`). Only DROP rows are removed from
discharge_daily.csv.gz; the untouched original is kept as discharge_daily_raw.csv.gz.
"""
import _bootstrap  # noqa: F401
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from floodcast import config

INTERIM, REPORTS = Path(config.INTERIM), Path(config.INTERIM).parent.parent / "reports"


def main():
    f = INTERIM / "discharge_daily.csv.gz"
    raw = INTERIM / "discharge_daily_raw.csv.gz"
    if not raw.exists():
        shutil.copy(f, raw)                      # always QC from the untouched original
    df = pd.read_csv(raw, parse_dates=["date"]).sort_values(["station_id", "date"]).reset_index(drop=True)

    flags = []
    for sid, g in df.groupby("station_id"):
        s = g.set_index("date")["discharge"]
        full = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
        p999 = full.quantile(0.999)
        # median of the 6 neighbouring days, excluding the day itself
        nb = pd.concat([full.shift(k) for k in (-3, -2, -1, 1, 2, 3)], axis=1).median(axis=1)
        r1 = (full > 20 * max(p999, 1e-9)) & (full > 0)
        r2 = (full > 30 * nb.clip(lower=1e-9)) & (full > 500)
        bad = full[(r1 | r2) & full.notna()]  # r1 => drop, r2 only => review
        for d, v in bad.items():
            flags.append({"station_id": sid, "date": d, "discharge": v, "station_p999": p999,
                          "neighbour_median": nb.loc[d], "rule1": bool(r1.loc[d]), "rule2": bool(r2.loc[d]),
                          "action": "drop" if r1.loc[d] else "review"})
    fl = pd.DataFrame(flags)
    REPORTS.mkdir(exist_ok=True)
    fl.to_csv(REPORTS / "qc_flags.csv", index=False)
    if fl.empty or not (fl.action == "drop").any():
        clean = df
    else:
        key = set(zip(fl.loc[fl.action == "drop", "station_id"], fl.loc[fl.action == "drop", "date"]))
        clean = df[[(a, b) not in key for a, b in zip(df.station_id, df.date)]]
    clean.to_csv(f, index=False)
    print(f"dropped {len(df) - len(clean)} rows, {int((fl.action == 'review').sum()) if len(fl) else 0} flagged for review; "
          f"max discharge now {clean.discharge.max():.0f}")
    if len(fl):
        out = fl.sort_values(["action", "discharge"], ascending=[True, False]).copy()
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        print(out.head(30).round(1).to_string(index=False))


if __name__ == "__main__":
    main()