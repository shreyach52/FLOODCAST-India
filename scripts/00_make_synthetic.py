"""Make FAKE CWC-style CSVs so the pipeline can be tested without the real data.
The header names below are guesses, NOT the real CWC headers. Output: data/raw/synthetic/
"""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from floodcast.config import DATA_RAW


def make(n_stations=12, start="2001-01-01", end="2010-12-31", seed=0):
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D")
    doy = days.dayofyear.to_numpy()
    rows = []
    for i in range(n_stations):
        base = rng.uniform(5, 50)
        amp = rng.uniform(100, 600)
        monsoon = np.exp(-((doy - 220) ** 2) / (2 * 35.0 ** 2))
        noise = rng.gamma(2.0, 0.3, size=len(days))
        q = base + amp * monsoon * noise
        keep = rng.random(len(days)) > rng.uniform(0.02, 0.25)   # random gaps
        for d, v in zip(days[keep], q[keep]):
            rows.append((f"KA{i:03d}", f"Station {i}", f"River {i % 3}", "Karnataka",
                         12.0 + i * 0.1, 75.0 + i * 0.1, d, round(float(v), 3)))
    return pd.DataFrame(rows, columns=["code", "name", "river", "state", "lat", "lon",
                                       "date", "q"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA_RAW / "synthetic"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    full = make()
    # file 1: ISO dates, one style of header
    p1 = full[full["date"] < "2006-01-01"].rename(columns={
        "code": "Station Code", "name": "Station Name", "river": "River", "state": "State",
        "lat": "Latitude", "lon": "Longitude", "date": "Date", "q": "Discharge (cumecs)"})
    p1.to_csv(out / "synthetic_a_2001_2005.csv", index=False)
    # file 2: different header names + dd-mm-yyyy dates, to test the guesser
    p2 = full[full["date"] >= "2006-01-01"].copy()
    p2["date"] = p2["date"].dt.strftime("%d-%m-%Y")
    p2 = p2.rename(columns={"code": "STATION_ID", "name": "STATION_NAME", "river": "RIVER_NAME",
                            "state": "STATE_NAME", "lat": "LAT", "lon": "LON",
                            "date": "OBS_DATE", "q": "DISCHARGE"})
    p2.to_csv(out / "synthetic_b_2006_2010.csv", index=False)
    print(f"Wrote 2 fake files to {out}")
