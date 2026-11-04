"""Read raw CWC CSVs and turn them into the canonical table defined in contracts.py.

The real column names are not known until the files are opened, so:
  1) guess_column_map() proposes a mapping from the header,
  2) the team checks/edits configs/column_map.json (one-time job),
  3) canonicalize() applies the map and cleans the data.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import (CANON_OPTIONAL, CANON_REQUIRED, CFS_PER_M3S, DATE, STATION,
                        TARGET)

# canonical name -> header keywords, strongest first.
# Short keywords (<=4 chars) must match a whole word; longer ones match as substrings.
RULES = {
    DATE: ["date", "datetime", "timestamp", "observation", "time"],
    TARGET: ["discharge", "cumec", "flow", "q"],
    STATION: ["station code", "stationcode", "station id", "stn code", "site code",
              "station", "site", "code"],
    "station_name": ["station name", "site name", "name"],
    "river": ["river"],
    "state": ["state"],
    "lat": ["latitude", "lat"],
    "lon": ["longitude", "lon", "long"],
}
ASSIGN_ORDER = [DATE, TARGET, STATION, "station_name", "river", "state", "lat", "lon"]


def _norm(c) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(c).lower()).strip()


def _matches(key: str, norm_col: str) -> bool:
    if len(key) <= 4:
        return key in norm_col.split()
    return key in norm_col


def guess_column_map(columns) -> dict:
    """Return {canonical_name: raw_column}. Each raw column is used at most once."""
    used, mapping = set(), {}
    norm = {c: _norm(c) for c in columns}
    for canon in ASSIGN_ORDER:
        found = None
        for key in RULES[canon]:
            for col in columns:
                if col not in used and _matches(key, norm[col]):
                    found = col
                    break
            if found:
                break
        if found:
            mapping[canon] = found
            used.add(found)
    return mapping


def load_column_map(path: Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_raw(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Read a CSV as text (so nothing is silently mangled)."""
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig", nrows=nrows,
                       skipinitialspace=True, low_memory=False)


def parse_dates(s: pd.Series) -> tuple[pd.Series, str]:
    """Parse dates WITHOUT letting pandas guess per row (that silently swaps day/month).

    Numeric dates like 05-06-2001 are decoded explicitly:
      - year first (2001-06-05)            -> ISO
      - a value > 12 in the first slot      -> day-first
      - a value > 12 in the second slot     -> month-first
      - everything ambiguous                -> day-first ASSUMED (Indian convention), flagged
    Anything else (e.g. 05-Jun-2001) goes through pandas' text parser.
    Returns (dates, label). The label is printed in the report; read it.
    """
    s = s.astype(str).str.strip()
    p = s.str.extract(r"^(\d{1,4})[-/.](\d{1,4})[-/.](\d{1,4})").apply(pd.to_numeric, errors="coerce")
    ok = p.notna().all(axis=1)
    parts = []
    label = "text-parser"
    if ok.mean() > 0.5:
        a, b, c = p[0][ok], p[1][ok], p[2][ok]
        if (a > 31).mean() > 0.5:
            y, m, d, label = a, b, c, "year-first (ISO)"
        else:
            y = c
            a_big, b_big = int((a > 12).sum()), int((b > 12).sum())
            if a_big and not b_big:
                d, m, label = a, b, "day-first (detected)"
            elif b_big and not a_big:
                m, d, label = a, b, "month-first (detected)"
            elif a_big and b_big:
                if a_big >= b_big:
                    d, m = a, b
                else:
                    m, d = a, b
                label = "WARNING: conflicting day/month order, used the majority"
            else:
                d, m, label = a, b, "day-first (ASSUMED: every value ambiguous - verify!)"
        y = y.where(y >= 100, y + np.where(y < 50, 2000, 1900))
        parts.append(pd.to_datetime({"year": y, "month": m, "day": d}, errors="coerce"))
    rest = s[~ok]
    if len(rest):
        parts.append(pd.to_datetime(rest, errors="coerce", format="mixed"))
    out = pd.concat(parts).reindex(s.index)
    return out, label


def canonicalize(raw: pd.DataFrame, column_map: dict | None = None,
                 source_unit: str = "m3s") -> tuple[pd.DataFrame, dict]:
    """Apply the column map, clean, and return (canonical_df, stats)."""
    cmap = column_map if column_map and all(v in raw.columns for v in column_map.values()) \
        else guess_column_map(list(raw.columns))
    missing = [c for c in CANON_REQUIRED if c not in cmap]
    if missing:
        raise ValueError(f"Could not map required columns {missing}. Header was: "
                         f"{list(raw.columns)}. Edit configs/column_map.json.")
    if source_unit not in ("m3s", "cfs"):
        raise ValueError("source_unit must be 'm3s' or 'cfs'")

    df = pd.DataFrame({canon: raw[col] for canon, col in cmap.items()})
    stats = {"rows_in": len(df), "column_map": cmap}

    df[DATE], how = parse_dates(df[DATE])
    stats["date_parse"] = how
    df[STATION] = df[STATION].astype(str).str.strip()
    df[TARGET] = pd.to_numeric(df[TARGET].astype(str).str.replace(",", "", regex=False)
                               .str.strip(), errors="coerce")

    stats["bad_dates"] = int(df[DATE].isna().sum())
    stats["bad_station"] = int((df[STATION].isin(["", "nan", "None"])).sum())
    stats["bad_discharge"] = int(df[TARGET].isna().sum())
    stats["negative_discharge"] = int((df[TARGET] < 0).sum())
    df.loc[df[TARGET] < 0, TARGET] = np.nan

    df = df[df[DATE].notna() & ~df[STATION].isin(["", "nan", "None"]) & df[TARGET].notna()]
    df[DATE] = df[DATE].dt.normalize()
    if source_unit == "cfs":
        df[TARGET] = df[TARGET] / CFS_PER_M3S

    for opt in ("lat", "lon"):
        if opt in df.columns:
            df[opt] = pd.to_numeric(df[opt], errors="coerce")

    # merge duplicate (station, day) rows by their mean; keep first value of text columns
    n_before = len(df)
    agg = {c: "first" for c in df.columns if c not in (STATION, DATE)}
    agg[TARGET] = "mean"
    df = df.groupby([STATION, DATE], as_index=False).agg(agg)
    stats["duplicates_merged"] = n_before - len(df)
    stats["rows_out"] = len(df)
    stats["n_stations"] = int(df[STATION].nunique())
    if stats["duplicates_merged"] > 0.2 * max(n_before, 1):
        import warnings
        warnings.warn(f"{stats['duplicates_merged']:,} of {n_before:,} rows were merged as duplicate "
                      f"(station, date) pairs but only {stats['n_stations']} station value(s) found. "
                      f"The station column in configs/column_map.json is probably wrong "
                      f"(mapped to {cmap[STATION]!r}).", stacklevel=2)

    ordered = [c for c in CANON_REQUIRED + CANON_OPTIONAL if c in df.columns]
    return df[ordered].sort_values([STATION, DATE]).reset_index(drop=True), stats


def station_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Per-station data quality: how many days, over what span, how many gaps."""
    d = df.sort_values([STATION, DATE]).copy()
    d["_gap"] = d.groupby(STATION)[DATE].diff().dt.days
    g = d.groupby(STATION)
    out = pd.DataFrame({
        "n_obs": g[DATE].size(),
        "first_date": g[DATE].min(),
        "last_date": g[DATE].max(),
        "max_missing_days_in_a_row": (g["_gap"].max().fillna(1) - 1).clip(lower=0),
        "q_median": g[TARGET].median(),
        "q_p99": g[TARGET].quantile(0.99),
        "q_max": g[TARGET].max(),
    })
    span = (out["last_date"] - out["first_date"]).dt.days + 1
    out["span_years"] = (span / 365.25).round(2)
    out["coverage"] = (out["n_obs"] / span).round(3)
    return out.reset_index()


def eligible_stations(cov: pd.DataFrame, min_years: float = 5.0,
                      min_coverage: float = 0.5) -> list:
    ok = cov[(cov["span_years"] >= min_years) & (cov["coverage"] >= min_coverage)]
    return ok[STATION].tolist()