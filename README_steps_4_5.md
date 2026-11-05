# Steps 4b + 5: threshold-free early-warning evaluation, and the value of future rain

Copy the folders into your project (nothing here overwrites your files; it only ADDS):
  floodcast/onset_eval.py  floodcast/futurerain.py
  scripts/08_onset_curves.py  scripts/09_future_rain_value.py
  tests/test_onset_eval.py  tests/test_cli_onset.py  tests/test_futurerain.py  tests/test_cli_future_rain.py
It imports your floodcast/evalcommon.py and floodcast/features_wx.py unchanged. Needs: lightgbm (already used).

Check:  python -m pytest -q            (the 4 new test files add 19 tests; the future-rain one takes ~30 s)

## 4b. Compare models without one alarm rule
  python scripts/08_onset_curves.py --pred <predictions file> --model lstm=<its prediction column> \
      --pred2 <lightgbm predictions> --model2 lightgbm_wx=<its column>
 Needs per row: station_id, date, horizon, split, q0, y, thr + prediction column(s).
 Output: reports/onset_curves.md, data/interim/onset_summary.csv

## 5. What is future rain worth?  (DIAGNOSTIC ONLY - uses rain that fell AFTER the forecast time)
  python scripts/09_future_rain_value.py --quick      # check it runs (about a minute)
  python scripts/09_future_rain_value.py              # full run
 Output: reports/future_rain_value.md, data/interim/future_rain_value.csv
 The oracle_* features must never be used by the real model, the dashboard or a submission.

What was verified: all tests, on SYNTHETIC data (a world where rain really drives the river). A --quick smoke run
on the older real files (partial weather, older QC) completed without errors. No real results are claimed.
