## Model Performance and Evaluation

FLOODCAST INDIA uses LightGBM regression models to forecast river discharge 1, 3, and 7 days ahead. Two configurations are evaluated:

- **LightGBM:** Uses historical river discharge, lagged values, rolling statistics, and station characteristics.
- **LightGBM + Weather:** Adds ERA5-Land rainfall and soil-moisture features.

### Evaluation methodology

- Training period: Up to 31 December 2018.
- Validation period: Up to 31 December 2021.
- Testing period: From 1 January 2022 onward.
- Stations included: 98.
- Training stations: 63.
- Unseen evaluation stations: 35 (10 randomly held-out stations and 25 late-start stations).

Two evaluation groups are used:

- **Seen stations:** Stations represented in the training data, evaluated on future dates.
- **Unseen stations:** Stations excluded from training, including stations with records beginning after the training period.

### Evaluation metrics

| Metric | Meaning |
|---|---|
| RMSE | Overall root mean squared error; penalizes large forecast errors. |
| Extreme RMSE | RMSE calculated only on actual high-flow days. |
| MAE | Mean absolute error. |
| R² | Coefficient of determination; measures explained variance. |

Extreme-flow days are defined using each station's 95th-percentile discharge threshold calculated from its training-period reference data.

### Test results

Results below compare the discharge-only model with the weather-enhanced model.

| Horizon | Station group | Model | RMSE | Extreme RMSE | MAE | R² |
|---|---|---|---:|---:|---:|---:|
| 1 day | Seen | LightGBM | 115.955 | 317.700 | 24.460 | 0.840 |
| 1 day | Seen | LightGBM + Weather | 116.006 | 320.458 | 24.007 | 0.839 |
| 1 day | Unseen | LightGBM | 144.035 | 449.293 | 24.188 | 0.792 |
| 1 day | Unseen | LightGBM + Weather | 129.065 | 399.814 | 22.683 | 0.833 |
| 3 days | Seen | LightGBM | 187.056 | 540.980 | 38.826 | 0.586 |
| 3 days | Seen | LightGBM + Weather | 185.450 | 534.859 | 37.931 | 0.593 |
| 3 days | Unseen | LightGBM | 234.617 | 745.587 | 38.931 | 0.449 |
| 3 days | Unseen | LightGBM + Weather | 231.833 | 736.130 | 38.578 | 0.462 |
| 7 days | Seen | LightGBM | 231.782 | 680.574 | 48.763 | 0.367 |
| 7 days | Seen | LightGBM + Weather | 223.871 | 654.259 | 47.748 | 0.410 |
| 7 days | Unseen | LightGBM | 269.730 | 850.819 | 48.732 | 0.267 |
| 7 days | Unseen | LightGBM + Weather | 268.635 | 848.069 | 48.255 | 0.273 |

*Discharge error units follow the units provided by the source CWC data.*

### Key findings

- Weather features improve all four reported metrics for unseen 1-day forecasts.
- Weather improves overall RMSE, extreme RMSE, MAE, and R² for both seen and unseen 3-day forecasts.
- Weather improves all four metrics for seen 7-day forecasts.
- Improvements for unseen 7-day forecasts are smaller, indicating that longer-range forecasting remains challenging.
- The 1-day seen-station model shows a small trade-off: weather improves MAE but slightly worsens RMSE, extreme RMSE, and R².

### Limitations

- Forecast performance varies across stations and forecast horizons.
- Some stations have limited historical discharge records.
- Unseen stations are more challenging because the model has no station-specific training history.
- Forecasts are experimental estimates and should not be treated as official flood warnings.
- Further validation with additional stations, hydrological features, and operational forecasting data is required.

These results represent the current prototype's held-out test evaluation, not a guarantee of real-world flood prediction accuracy.