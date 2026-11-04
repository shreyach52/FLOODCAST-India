"""Pure helpers for the ERA5-Land weather join (numpy/pandas only, so they are unit-testable)."""
from __future__ import annotations

import calendar
from datetime import date, timedelta

import numpy as np

DATASET = "reanalysis-era5-land"
# CDS area order is [North, West, South, East]. Covers all 107 stations (lat 8.42-18.05, lon 74.53-77.47).
AREA = [18.5, 74.0, 8.0, 78.0]
VARIABLES = [
    "total_precipitation",
    "volumetric_soil_water_layer_1"
]
# Only the 00:00 UTC field of each day is requested (~365 fields per variable-year, tiny download).
# ERA5-Land precipitation is accumulated from 00 UTC, so the 00:00 value of day D holds the 24 h total of
# the PREVIOUS 24 h (D-1 00:00 -> D 00:00 UTC = 05:30 IST). We label it with date D: it is known before
# the morning discharge reading of day D, so using it as a feature on day D does not look into the future.
TIME = ["00:00"]


def plan_chunks(first_year: int = 2015, today: date | None = None, lag_days: int = 10):
    """List of (tag, year, [months]) requests. Whole years for finished years; for the current year only
    months that ended at least `lag_days` ago (ERA5-Land is released a few days behind real time)."""
    today = today or date.today()
    out = []
    for y in range(first_year, today.year + 1):
        if y < today.year:
            out.append((f"{y}", y, list(range(1, 13))))
        else:
            ok = [m for m in range(1, 13)
                  if date(y, m, calendar.monthrange(y, m)[1]) <= today - timedelta(days=lag_days)]
            out.extend((f"{y}_{m:02d}", y, [m]) for m in ok)
    return out


def build_request(year: int, months: list[int]) -> dict:
    return {
        "variable": VARIABLES,
        "year": [str(year)],
        "month": [f"{m:02d}" for m in months],
        "day": [f"{d:02d}" for d in range(1, 32)],   # invalid dates (e.g. 31 Feb) are ignored by CDS
        "time": TIME,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": AREA,
    }


def nearest_valid_cells(st_lat, st_lon, grid_lat, grid_lon, valid):
    """For each station return (iy, ix, distance_km) of the nearest grid cell where `valid[iy, ix]` is True.
    `valid` is the land mask (False over sea / missing), so coastal stations never land on a NaN cell."""
    st_lat, st_lon = np.asarray(st_lat, float), np.asarray(st_lon, float)
    iy_all, ix_all = np.nonzero(valid)
    cl, cn = np.asarray(grid_lat)[iy_all], np.asarray(grid_lon)[ix_all]
    iy, ix, km = [], [], []
    for la, lo in zip(st_lat, st_lon):
        dy = np.radians(cl - la)
        dx = np.radians(cn - lo) * np.cos(np.radians((cl + la) / 2))
        d = 6371.0 * np.sqrt(dx ** 2 + dy ** 2)
        k = int(np.argmin(d))
        iy.append(int(iy_all[k])); ix.append(int(ix_all[k])); km.append(float(d[k]))
    return np.array(iy), np.array(ix), np.array(km)