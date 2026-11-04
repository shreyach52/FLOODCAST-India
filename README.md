# FLOODCAST INDIA - Step 1: data layer

Goal of this step: get the real CWC river-discharge data onto your machine, look at its
real columns, and turn it into ONE clean table (`station_id, date, discharge`) that every
later step uses. Nothing here trains a model yet.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                # should print: 27 passed
```

## Try the pipeline on FAKE data first (30 seconds)
```bash
python scripts/00_make_synthetic.py
python scripts/02_inspect_schema.py --dir data/raw/synthetic --map-out configs/column_map_synthetic.json
python scripts/03_build_discharge_table.py --dir data/raw/synthetic --out-dir data/interim_synthetic --map configs/column_map_synthetic.json
```

## Run it on the REAL data
```bash
# 1. download (start small: one or two states)
python scripts/01_download_cwc.py --states KA KL
#    other eras:  --eras 1950_2000 2001_2025 2026_2030      all states: --all

# 2. look at the real headers (prints columns + sample rows + a guessed column map)
python scripts/02_inspect_schema.py
#    -> open configs/column_map.json and FIX anything that is wrong

# 3. clean + merge + data-quality report
python scripts/03_build_discharge_table.py
#    if the raw numbers are in cubic feet per second:  add  --source-unit cfs
```
Outputs: `data/interim/discharge_daily.csv.gz`, `data/interim/station_coverage.csv`,
`reports/data_report.md`. READ the report: it decides which models are possible.

If a download fails, open the link in a browser, save the CSV into `data/raw/cwc/`, and continue.

## Files
- `floodcast/contracts.py`  FROZEN interfaces (column names, units, validators, splits). Read `docs/TEAM_CONTRACTS.md`.
- `floodcast/io_cwc.py`     header guesser, date parser, cleaner, station coverage.
- `floodcast/config.py`     paths + CWC download registry.
- `docs/DATA_SOURCES.md`    all six data sources: what each gives, how to get it, what is verified.
- `docs/STEP_LOG.md`        what is done and what comes next.
