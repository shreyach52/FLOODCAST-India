# Step log

## Step 1 - data layer (DONE in this zip)
- Frozen contracts, CWC download registry (all 3 eras), header guesser, deterministic date parser,
  cleaner, per-station coverage report, synthetic data generator, 27 tests.
- NOT done: real data not downloaded or inspected yet (needs your internet); real column names unknown.

## Step 2 - look at the real data (next, needs your output)
Run scripts 01-03 on 1-2 states and send me: the printed columns + 3 sample rows per file,
`reports/data_report.md`. I will then fix the column map, decide units, and check for station lat/lon.

## Later
3. Rainfall + soil moisture joins   4. Baseline + persistence + metrics (NSE/KGE)
5. CatBoost/LightGBM   6. LSTM multi-horizon   7. CQR intervals + exceedance probability
8. Risk levels + SHAP explanations   9. Streamlit dashboard   10. README, slides, demo
