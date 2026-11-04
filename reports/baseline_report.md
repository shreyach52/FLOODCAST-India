# Baseline report (discharge-only)

Train <= 2018-12-31, val <= 2021-12-31, test >= 2022-01-01. Held-out stations (14): ['AYILAM', 'Airani', 'Cholachagudda', 'KOLLEGAL', 'KONDAZHY', 'LAKKAVALLI', 'MANKARA', 'NEELEESWARAM', 'Navalgund', 'PERUMANNU', 'PUDUR', 'PULLAKKAYAR', 'THORESHETTAHALLI', 'Talikot']

High flow = station's train-period 95th percentile. hi_recall = share of true high-flow days whose forecast also reached that level; hi_precision = share of such alarms that were real.

```
 horizon             split       model     n    rmse    mae  median_station_nse  hi_n  hi_rmse  hi_recall  hi_precision  onset_n  onset_recall  onset_alarms  onset_precision
       1   A_seen_stations persistence 95038 130.823 23.212               0.724 10445  315.994      0.885         0.884     1203         0.000             0              NaN
       1   A_seen_stations climatology 95038 269.335 50.675               0.302 10445  777.334      0.431         0.928     1203         0.006           158            0.044
       1   A_seen_stations    lightgbm 95038 147.602 26.914               0.732 10445  349.651      0.851         0.908     1203         0.068           235            0.349
       1 B_unseen_stations persistence 14048  99.259 24.511               0.671  2706  207.961      0.932         0.933      183         0.000             0              NaN
       1 B_unseen_stations climatology 14048 174.613 44.412               0.267  2706  383.729      0.731         0.987      183         0.000             7            0.000
       1 B_unseen_stations    lightgbm 14048 108.127 25.654               0.643  2706  233.684      0.917         0.958      183         0.011             9            0.222
       3   A_seen_stations persistence 94025 199.796 40.893               0.375 10304  492.076      0.795         0.792     2113         0.000             0              NaN
       3   A_seen_stations climatology 94025 272.709 53.466               0.264 10304  782.699      0.413         0.893     2113         0.009           164            0.110
       3   A_seen_stations    lightgbm 94025 207.139 40.971               0.456 10304  538.668      0.727         0.839     2113         0.063           510            0.263
       3 B_unseen_stations persistence 13838 165.851 43.275               0.199  2675  348.335      0.886         0.883      305         0.000             0              NaN
       3 B_unseen_stations climatology 13838 197.728 51.490               0.130  2675  436.271      0.722         0.972      305         0.003             9            0.111
       3 B_unseen_stations    lightgbm 13838 164.794 40.841               0.287  2675  360.679      0.855         0.939      305         0.020            13            0.462
       7   A_seen_stations persistence 96610 269.169 57.236              -0.017 10737  641.340      0.703         0.699     3194         0.000             0              NaN
       7   A_seen_stations climatology 96610 274.154 56.066               0.037 10737  776.923      0.398         0.854     3194         0.011           168            0.202
       7   A_seen_stations    lightgbm 96610 245.822 51.451               0.259 10737  678.356      0.606         0.793     3194         0.081           848            0.307
       7 B_unseen_stations persistence 14262 223.627 60.532              -0.212  2785  449.745      0.851         0.844      416         0.000             0              NaN
       7 B_unseen_stations climatology 14262 222.070 58.082              -0.008  2785  489.815      0.719         0.964      416         0.002             7            0.143
       7 B_unseen_stations    lightgbm 14262 202.522 52.674               0.076  2785  443.091      0.806         0.932      416         0.019            42            0.190
```