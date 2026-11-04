"""STEP 3b: turn the ERA5-Land NetCDF files into one daily table per station.

  python scripts/06b_extract_station_weather.py
Output: data/interim/weather_daily.csv.gz (station_id, date, rain_mm, soil1, soil2, temp_c, cell_lat, cell_lon)
        reports/weather_report.md (sanity checks: read it!)
Each station uses the nearest LAND cell (coastal stations never fall on a sea/NaN cell).
rain_mm for date D = precipitation over the 24 h ending 00:00 UTC of D (known before D's morning reading).
"""
import _bootstrap  # noqa: F401
from pathlib import Path

import numpy as np
import pandas as pd

from floodcast import config
from floodcast.era5 import common_variables, nearest_valid_cells

RAW = Path(config.DATA_RAW) / "era5land"


def open_chunk(paths):
    import xarray as xr
    ds = xr.merge([xr.open_dataset(p) for p in paths], compat="override")
    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    if "expver" in ds.dims:                              # ERA5T (recent) vs final data
        ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))
    return ds.drop_vars([v for v in ("expver", "number") if v in ds.coords], errors="ignore")


def main():
    import xarray as xr
    stn = (pd.read_csv(Path(config.INTERIM) / "discharge_daily.csv.gz", usecols=["station_id", "lat", "lon"])
           .drop_duplicates("station_id").reset_index(drop=True))
    chunks = sorted([p for p in RAW.glob("era5land_*.nc")] + [p for p in RAW.glob("era5land_*") if p.is_dir()])
    if not chunks:
        raise SystemExit(f"no files in {RAW}; run 06a_download_era5land.py first")
    parts, cells = [], None
    opened = [(c, open_chunk([c] if c.is_file() else sorted(c.glob("*.nc")))) for c in chunks]
    used, dropped = common_variables([list(ds.data_vars) for _, ds in opened])
    print("variables used:", used, "| dropped (not in every chunk):", dropped or "none")
    for c, ds in opened:
        if cells is None:
            land = np.isfinite(ds["swvl1"].isel(time=0).values)
            lat, lon = ds["latitude"].values, ds["longitude"].values
            iy, ix, km = nearest_valid_cells(stn.lat, stn.lon, lat, lon, land)
            cells = (iy, ix, km, lat, lon)
        iy, ix = cells[0], cells[1]
        pts = ds.isel(latitude=xr.DataArray(iy, dims="st"), longitude=xr.DataArray(ix, dims="st"))
        t = pd.to_datetime(pts["time"].values).floor("D")
        for k, sid in enumerate(stn.station_id):
            parts.append(pd.DataFrame({
                "station_id": sid, "date": t,
                "rain_mm": np.clip(pts["tp"].values[:, k] * 1000.0, 0, None),   # m -> mm
                "soil1": pts["swvl1"].values[:, k],
                **({"soil2": pts["swvl2"].values[:, k]} if "swvl2" in used else {}),
                **({"temp_c": pts["t2m"].values[:, k] - 273.15} if "t2m" in used else {}),
                "cell_lat": cells[3][iy[k]], "cell_lon": cells[4][ix[k]]}))
        print("read", c.name, flush=True)
    w = pd.concat(parts, ignore_index=True).drop_duplicates(["station_id", "date"]).sort_values(["station_id", "date"])
    w.to_csv(Path(config.INTERIM) / "weather_daily.csv.gz", index=False)

    # sanity report
    ann = (w.assign(year=w.date.dt.year).groupby(["station_id", "year"]).rain_mm.sum()
           .groupby("station_id").median().rename("median_annual_rain_mm"))
    shared = w.drop_duplicates("station_id").groupby(["cell_lat", "cell_lon"]).size()
    days = w.drop_duplicates(["station_id", "date"]).groupby(w.date.dt.year).date.nunique()
    one = w.station_id.iloc[0]
    per_year = w[w.station_id == one].groupby(w.date.dt.year).size()
    missing_years = [y for y in range(int(per_year.index.min()), int(per_year.index.max()) + 1)
                     if y not in per_year.index]
    lines = ["# Weather report (ERA5-Land)\n",
             f"Variables used: {used}. Dropped because not in every chunk: {dropped or 'none'}.",
             f"Days per year (one station): {per_year.to_dict()}",
             f"**MISSING YEARS: {missing_years}**" if missing_years else "No missing years.",
             f"Rows {len(w):,}, stations {w.station_id.nunique()}, dates {w.date.min().date()} to {w.date.max().date()}.",
             f"Nearest-land-cell distance (km): median {np.median(cells[2]):.1f}, max {cells[2].max():.1f}.",
             f"Stations sharing a grid cell with another station: {int((shared > 1).sum())} cells hold {int(shared[shared > 1].sum())} stations.\n",
             "Median ANNUAL rainfall per station (mm). Western Ghats / coastal should be roughly 2000-4500, "
             "interior Deccan roughly 500-900. If everything is ~365x too small/large, the 24 h accumulation "
             "assumption is wrong - tell me.\n", "```", ann.describe().round(0).to_string(),
             "", ann.sort_values().head(5).round(0).to_string(), "", ann.sort_values().tail(5).round(0).to_string(), "```"]
    (Path(config.REPORTS) / "weather_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()