# Gap report (lookback=14d, horizons 1/3/7, min windows=500)

- Stations: **107**; sequence-ready: **99**; seasonal-only (<12 months present): **5**
- Total usable windows in ready stations: **378,732**

## Not sequence-ready (bottom 25 by windows)

```
                       n_obs  span_years  longest_run_days  gaps_over_30d  windows_1_3_7  months_present  monsoon_share   q_max  q_median    cv  sequence_ready  seasonal_only
station_id                                                                                                                                                                    
Deosugur                   1        0.00                 1              0              0               1           0.00   33.68     33.68   NaN           False           True
Hemavathy Reservoir        2        0.01                 1              0              0               1           1.00   20.12     18.85  0.10           False           True
OLAMATTOM                 26        0.08                 6              0              0               1           1.00  131.85     51.47  0.42           False           True
Narayanapur Reservoir      3        0.02                 1              0              0               1           0.00    2.42      2.35  0.12           False           True
Gokak                    241        0.67               208              0            203               8           0.37  433.66      0.00  2.35           False           True
PUDUNAGARA               656        5.04               183              7            310              12           0.41    8.19      0.26  1.67           False          False
SANOOR                  1420        8.46               151              7            366              12           0.65  127.40      2.96  1.64           False          False
KALATHUKADAVU           1720        7.19               307              6            458              12           0.49  188.32      4.71  1.30           False          False
```

## Sequence-ready summary

```
         n_obs  span_years  longest_run_days  gaps_over_30d  windows_1_3_7  months_present  monsoon_share      q_max  q_median     cv
count    99.00       99.00             99.00          99.00          99.00            99.0          99.00      99.00     99.00  99.00
mean   5252.71       18.43           1264.17           6.37        3825.58            12.0           0.44   15633.89     18.57   3.60
std    2907.75        8.78           1002.45           5.35        2479.28             0.0           0.05   87480.15     25.95   3.88
min     804.00        2.20            151.00           0.00         543.00            12.0           0.34      85.36      0.00   0.96
25%    2200.00        7.58            407.50           2.50        1200.50            12.0           0.41     565.30      2.46   1.78
50%    6433.00       24.88           1142.00           5.00        4128.00            12.0           0.43    1291.28      9.69   2.20
75%    8077.50       25.73           1705.50           9.00        6081.50            12.0           0.46    3329.23     24.30   3.82
max    9243.00       25.84           7091.00          22.00        8487.00            12.0           0.60  850891.00    135.03  31.24
```