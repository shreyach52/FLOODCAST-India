# Data sources

Legend: **checked** = I opened the page and read it. **not opened** = taken from the link only; verify yourself.

| # | Source | Gives us | Access | Status |
|---|---|---|---|---|
| 1 | CWC daily river discharge (National Water Data Portal) | THE TARGET: daily discharge per station | Direct CSV links, one file per state per era (1950-2000, 2001-2025, 2026-2030). `scripts/01_download_cwc.py` | **checked** (page + resource list). Column names NOT seen yet |
| 2 | NIH Roorkee daily discharge (same portal) | Extra stations / cross-check | Same portal | **not opened** |
| 3 | IMD gridded rainfall, 0.25 deg | Daily rainfall (mm) at 0.25 deg grid, 1901-2024 per the page (the year list also shows 2025) | Pick a year on the page, download NetCDF (manual) | **checked** |
| 4 | ERA5-Land (Copernicus CDS) | Soil moisture (4 layers), temperature, runoff etc., hourly, 0.1 deg, 1950-present | CDS request; page shows Login/Register, so expect a free account | **checked** (page only; download not tried) |
| 5 | NRSC land use / land cover atlas | Static catchment land cover | Web atlas | **not opened** |
| 6 | NASA SRTM | Elevation / slope for stations and catchments | Usually needs a free Earthdata login | **not opened** |

Facts from the CWC page: records have a date, a station identifier with geographic hierarchy, and
discharge values; both telemetric and manual measurements are included. Dataset last updated
Nov 2026 on the portal.

## What is NOT in the data yet (important)
- CWC discharge has no rainfall, soil moisture, temperature, or catchment attributes. Those must be
  joined from sources 3-6 using station latitude/longitude. If the CWC file has no lat/lon, we need a
  station-location table (the portal's River Network / India-WRIS pages may help). Step 2 checks this.
- Upstream/downstream links are NOT in the CWC file. Treat any link we infer as "inferred", never "verified".

## Join plan (Step 3+)
station (lat, lon) -> nearest IMD 0.25 deg cell (rainfall) and nearest ERA5-Land cell (soil moisture, temperature)
-> static features from SRTM / LULC -> one modelling table keyed by (station_id, date).
