# Step log

## DONE
1. Data layer: frozen contracts, CWC download (3 eras), header guesser, cleaner, coverage report. (`01`-`03`)
2. Gap analysis: 107 stations (KA 66, KL 41), 99 sequence-ready. (`04`)
3. QC of impossible discharge values, run-aware rules in `floodcast/qc.py`, human overrides via
   `configs/qc_decisions.csv`. (`03b`)
4. Discharge-only baseline: persistence, climatology, LightGBM (residual target), onset metrics. (`05`)
5. ERA5-Land weather 2001-2026 for every station (rain, soil layer 1), nearest land cell, no-leakage date
   convention. Annual rainfall checked against geography (665 mm interior - 3,843 mm Ghats). (`06a`, `06b`)

6. Weather features (`floodcast/features_wx.py`) added to the baseline; `05` reports lightgbm vs lightgbm_wx.
   Finding (discharge only): onset recall 6-8 % (seen) and 1-2 % (unseen) -> weather is required.
7. Evaluation fix: 25 stations (mostly Kerala) have NO training-period data (records start mid-2019). They used to be
   scored as "seen" with an empty threshold. They are now "late-start" unseen stations (reference stats from data up to
   val end). Shared logic: `floodcast/evalcommon.py::station_groups`. Old baseline numbers are NOT comparable; re-run 05.
8. LSTM (`floodcast/seq.py`, `scripts/07_lstm.py`): one network, 3 heads (1/3/7 d), residual target, same splits/metrics.

## IN PROGRESS
9. Run 05 (new split) then 07 --quick, then 07; compare lstm vs lightgbm_wx in `reports/lstm_report.md`.

## NEXT
10. Decide LSTM vs LightGBM per horizon (and --hi-weight 3, --seeds 3, --no-weather ablation).
11. Review remaining large values (T. NARASIPUR, MAHISHI, Huvinhedigi) -> `configs/qc_decisions.csv`.
12. Optional: IMD rain swap-in, CatBoost.  13. Quantile intervals.  14. Risk levels + SHAP.
15. Streamlit dashboard + map.  16. README, slides, demo.

## Known limits (write these in the final report)
- No forecast rain is used, so 3/7-day forecasts see only rain that has already fallen.
- Held-out stations are a random 15 %; reservoir/seasonal stations are excluded (regulated flow).
- No upstream/downstream connectivity map yet, so upstream features are NOT claimed.
