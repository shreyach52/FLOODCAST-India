"""STEP 2b: flag and remove impossible discharge values BEFORE modelling (rules: floodcast/qc.py).

  python scripts/03b_qc_outliers.py
Writes reports/qc_flags.csv (column `action`: drop / review). DROP rows are removed from
data/interim/discharge_daily.csv.gz; the untouched original is kept as discharge_daily_raw.csv.gz.
Always re-runs from the raw backup, so it is safe to run again after changing a rule or decisions.

Human override: put rows in configs/qc_decisions.csv (columns station_id,date,action with action = keep or drop).
  keep -> a flagged row is restored,  drop -> an unflagged (e.g. 'review') row is removed.
"""
import _bootstrap  # noqa: F401
import shutil
from pathlib import Path

import pandas as pd

from floodcast import config
from floodcast.qc import flag_station

INTERIM, REPORTS = Path(config.INTERIM), Path(config.REPORTS)
DECISIONS = Path(config.CONFIGS) / "qc_decisions.csv"


def main():
    f, raw = INTERIM / "discharge_daily.csv.gz", INTERIM / "discharge_daily_raw.csv.gz"
    if not raw.exists():
        shutil.copy(f, raw)
    df = pd.read_csv(raw, parse_dates=["date"]).sort_values(["station_id", "date"]).reset_index(drop=True)

    parts = []
    for sid, g in df.groupby("station_id"):
        fl = flag_station(g.set_index("date")["discharge"])
        if len(fl):
            fl.insert(0, "station_id", sid)
            parts.append(fl)
    fl = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["station_id", "date", "discharge", "action", "rule"])

    if DECISIONS.exists():                               # human decisions win over the rules
        dec = pd.read_csv(DECISIONS, parse_dates=["date"])
        for _, r in dec.iterrows():
            m = (fl.station_id == r.station_id) & (fl.date == r.date)
            if m.any():
                fl.loc[m, "action"] = r.action
                fl.loc[m, "rule"] = "human_decision"
            else:
                v = df.loc[(df.station_id == r.station_id) & (df.date == r.date), "discharge"]
                fl.loc[len(fl)] = [r.station_id, r.date, float(v.iloc[0]) if len(v) else float("nan"),
                                   r.action, "human_decision"]
    REPORTS.mkdir(exist_ok=True)
    fl.to_csv(REPORTS / "qc_flags.csv", index=False)

    drop = set(zip(fl.loc[fl.action == "drop", "station_id"], fl.loc[fl.action == "drop", "date"]))
    clean = df[[(a, b) not in drop for a, b in zip(df.station_id, df.date)]]
    clean.to_csv(f, index=False)
    nrev = int((fl.action == "review").sum())
    print(f"dropped {len(drop)} rows, {nrev} left in the data for review; "
          f"max discharge now {clean.discharge.max():.0f}; stations touched: {fl.station_id.nunique() if len(fl) else 0}")
    if len(fl):
        out = fl.sort_values(["action", "discharge"], ascending=[True, False]).copy()
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        print(out.head(40).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
