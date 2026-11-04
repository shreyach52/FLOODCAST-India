"""STEP 1b: look at the real headers and write a column map to check.

  python scripts/02_inspect_schema.py                 # reads data/raw/cwc
  python scripts/02_inspect_schema.py --dir data/raw/synthetic
It prints columns + sample rows for every CSV, proposes a column map, and writes
configs/column_map.json (only if it does not exist yet; use --overwrite to replace).
READ THE PRINTED MAP and fix configs/column_map.json by hand if anything is wrong.
"""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

import pandas as pd

from floodcast import config
from floodcast.io_cwc import guess_column_map, read_raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(config.RAW_CWC))
    ap.add_argument("--map-out", default=str(config.COLUMN_MAP_PATH))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--nrows", type=int, default=20000)
    a = ap.parse_args()

    files = sorted(Path(a.dir).glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files in {a.dir}. Run scripts/01_download_cwc.py first.")
    pd.set_option("display.width", 200, "display.max_columns", 30)

    first_map = None
    for f in files:
        df = read_raw(f, nrows=a.nrows)
        m = guess_column_map(list(df.columns))
        print("=" * 80)
        print(f"{f.name}   (read first {len(df)} rows)")
        print("columns:", list(df.columns))
        print(df.head(3).to_string(index=False))
        print("guessed map:", m)
        if first_map is None:
            first_map = m

    out = Path(a.map_out)
    if out.exists() and not a.overwrite:
        print(f"\n{out} already exists, not touching it (use --overwrite to replace).")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(first_map, indent=2), encoding="utf-8")
        print(f"\nWrote proposed map (from the first file) to {out}. CHECK IT.")


if __name__ == "__main__":
    main()
