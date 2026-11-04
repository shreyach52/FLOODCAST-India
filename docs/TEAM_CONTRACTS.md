# Frozen interfaces (v1)

Rule: import names from `floodcast/contracts.py`. Do not hard-code column names or units.
Change the contract only after telling the whole team, and bump the version in its docstring.

| Contract | Value |
|---|---|
| Canonical table | one row per (station, day): `station_id, date, discharge` (+ optional `station_name, river, state, lat, lon`) |
| Canonical unit | m3/s (`UNIT = "m3s"`). Convert at the edges only (`cfs_to_m3s`, `m3s_to_cfs`) |
| Horizons | 1, 3, 7 days |
| Forecast output | DataFrame with `station_id, issue_date, horizon, q_pred, q_low, q_high`; must pass `validate_forecast_df` |
| Risk levels | Normal, Elevated, High, Extreme |
| Splits | always `make_splits()`: chronological train/val/test on seen stations + held-out "ungauged" stations |
| Seed | 42 |

Results log: one CSV row per experiment in `reports/results.csv` with
`model, horizon, split, rmse, mae, nse, kge`-style metrics (exact columns agreed in Step 4).

Suggested owners (edit for your team size): A data/features, B tabular models,
C sequence + uncertainty, D risk logic + dashboard + slides.
