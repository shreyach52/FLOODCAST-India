"""STEP 3a: download ERA5-Land daily fields (rain, soil moisture 0-7 & 7-28 cm, 2 m temperature) for Karnataka+Kerala.

One-time setup (free):
  1. Register at https://cds.climate.copernicus.eu and log in.
  2. Open the ERA5-Land dataset page (reanalysis-era5-land) -> Download tab -> accept the licence.
  3. Copy your personal access token from your CDS profile and create  %USERPROFILE%\\.cdsapirc  (Windows) or ~/.cdsapirc :
         url: https://cds.climate.copernicus.eu/api
         key: <YOUR-PERSONAL-ACCESS-TOKEN>
  4. pip install cdsapi xarray netCDF4
Run:  python scripts/06a_download_era5land.py          (safe to re-run: finished chunks are skipped)
"""
import _bootstrap  # noqa: F401
import zipfile
from pathlib import Path

from floodcast import config
from floodcast.era5 import DATASET, build_request, plan_chunks

OUT = Path(config.DATA_RAW) / "era5land"


def main():
    import cdsapi
    OUT.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    chunks = plan_chunks()
    print(f"{len(chunks)} requests planned -> {OUT}")
    failed = []
    for tag, year, months in chunks:
        target, folder = OUT / f"era5land_{tag}.nc", OUT / f"era5land_{tag}"
        if target.exists() or (folder.exists() and any(folder.glob("*.nc"))):
            print("skip", tag); continue
        print("requesting", tag, flush=True)
        try:
            tmp = OUT / f"era5land_{tag}.part"
            client.retrieve(DATASET, build_request(year, months)).download(str(tmp))
            if zipfile.is_zipfile(tmp):                 # CDS sometimes returns a zip of per-variable files
                folder.mkdir(exist_ok=True)
                with zipfile.ZipFile(tmp) as z:
                    z.extractall(folder)
                tmp.unlink()
            else:
                tmp.rename(target)
        except Exception as e:                           # keep going; re-run later to retry
            print("FAILED", tag, repr(e)); failed.append(tag)
    print("done. failed:", failed or "none")


if __name__ == "__main__":
    main()