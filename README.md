# FLOODCAST INDIA

Goal of this step: get the real CWC river-discharge data onto your machine, look at its
real columns, and turn it into ONE clean table (`station_id, date, discharge`) that every
later step uses. Nothing here trains a model yet.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                # all tests should pass
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


## Pipeline after step 1 (run in this order)

```
python scripts/04_gap_analysis.py
python scripts/03b_qc_outliers.py              # drops impossible discharge values, writes reports/qc_flags.csv
python scripts/06a_download_era5land.py        # needs a free Copernicus CDS account + ~/.cdsapirc (see script header)
python scripts/06b_extract_station_weather.py  # -> data/interim/weather_daily.csv.gz, reports/weather_report.md
python scripts/05_baseline.py                  # -> reports/baseline_report.md (with vs without weather)
python scripts/07_lstm.py --quick              # 2-min pipeline check, then:  python scripts/07_lstm.py
```

Status and next steps: `docs/STEP_LOG.md`. Large data folders are git-ignored on purpose.
