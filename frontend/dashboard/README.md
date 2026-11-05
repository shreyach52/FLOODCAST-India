# FLOODCAST INDIA dashboard (frontend only)

Run (Windows 11, from the project root, same venv as the backend; no extra packages):

    python dashboard\server.py        then open http://127.0.0.1:8000     (or double-click run_dashboard.bat)

The server is read-only. It reads files your backend already wrote and shows "unavailable" for anything missing.

| Screen part | Reads | If missing |
|---|---|---|
| Stations, history, thresholds, risk | data/interim/discharge_daily.csv.gz | server stops with a clear message |
| Actual vs predicted, 1/3/7-day cards | data/processed/forecasts.csv.gz | "Predictions unavailable", observed only |
| Weather charts | data/interim/weather_daily.csv.gz | "Weather unavailable" |
| Model performance | files listed in `metrics_files` | "Metrics unavailable" |

All paths and risk settings are in configs/frontend.json. If your LSTM metrics or forecast files have other names,
edit that file only. Metrics CSVs need the baseline_metrics.csv columns (horizon, split, model, rmse, mae,
median_station_nse, hi_*, onset_*), with model = "lstm" for the LSTM rows.

## Forecast file contract (the existing FORECAST_COLS)
station_id, issue_date, horizon, q_pred, q_low, q_high  (q_low/q_high may be empty; optional `split` column).
A forecast issued on D with horizon h is plotted at D+h against the observation of D+h; dates after a station's
last observation are shaded as the forecast period.

## Smallest hook if your scripts do not save predictions yet
In scripts/05_baseline.py (after `te["p_lgb"]` is computed) collect and, after the loop, write:
    fc_rows.append(te[["station_id","date"]].rename(columns={"date":"issue_date"}).assign(horizon=h, q_pred=te.p_lgb, split=name))
    ...
    out = pd.concat(fc_rows); out["q_low"]=out["q_high"]=np.nan
    (config.ROOT/"data"/"processed").mkdir(exist_ok=True); out.to_csv(config.ROOT/"data"/"processed"/"forecasts.csv.gz", index=False)
This saves what the model already predicted; it changes no modelling logic.

## Risk levels (indicative)
High = latest observed discharge >= station's 95th percentile up to train_end (the backend's high-flow definition).
Elevated = >= 90th percentile (configured, editable). Only stations reporting within `fresh_days` are classified.
