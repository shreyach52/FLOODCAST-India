# LSTM report (discharge + weather)

Window 60 d, hidden 64 x 2 layers, hi-weight 1.0, seeds 1, epochs<= 12. Train <= 2018-12-31, val <= 2021-12-31, test >= 2022-01-01. Unseen stations: 10 random held-out ['BANTWAL', 'BELUR', 'ERINJIPUZHA', 'KOKKEDODDY', 'MUTHANKERA', 'Nagalamedike', 'Navalgund', 'Saigaon', 'THORESHETTAHALLI', 'Yadgir'] + 25 late-start ['ADKASTHALA', 'ARKKANNUR', 'ARUVIPURAM', 'Airani', 'CHAKKALAKUTH', 'HARIHARAPURA', 'HOMMARAGALLI', 'JANWADA', 'KONDAZHY', 'KOTTAPURAM', 'Kalmane', 'LAKKAVALLI', 'MADAMON', 'MAHISHI', 'MANAKKAD', 'MANIKAL', 'NELLIPPALLI', 'PALAKADAVU', 'PALLIPPADY', 'PERUVAMPADAM', 'POOKKAYAM', 'PULIKUKKU', 'PULLAKKAYAR', 'THOTTATHINKADAVU', 'VELLAIKADAVU']

lightgbm rows come from data/interim/baseline_metrics.csv (re-run 05_baseline.py with the SAME flags first, otherwise they are not comparable).

```
 horizon             split       model     n    rmse  median_station_rmse  median_station_nse  hi_n  hi_rmse  hi_recall  hi_precision  onset_n  onset_recall  onset_alarms  onset_precision
       1   A_seen_stations persistence 67056 116.343               48.310               0.720  6965  298.597      0.843         0.842     1094         0.000             0              NaN
       1   A_seen_stations    lightgbm 67056 115.955               46.354               0.751  6965  317.700      0.796         0.879     1094         0.053           175            0.331
       1   A_seen_stations lightgbm_wx 67056 116.006               45.571               0.741  6965  320.458      0.801         0.876     1094         0.088           254            0.378
       1   A_seen_stations        lstm 67056 112.845               46.007               0.747  6965  305.861      0.814         0.874     1094         0.039           127            0.339
       1 B_unseen_stations persistence 40269 106.769               37.015               0.742  3814  314.398      0.845         0.841      592         0.000             0              NaN
       1 B_unseen_stations    lightgbm 40269 144.035               35.666               0.721  3814  449.293      0.799         0.858      592         0.071           112            0.375
       1 B_unseen_stations lightgbm_wx 40269 129.065               31.380               0.735  3814  399.814      0.811         0.846      592         0.111           195            0.338
       1 B_unseen_stations        lstm 40269 112.505               33.611               0.767  3814  344.204      0.815         0.862      592         0.052            76            0.408
       3   A_seen_stations persistence 66239 189.526               81.537               0.375  6866  482.587      0.728         0.725     1867         0.000             0              NaN
       3   A_seen_stations    lightgbm 66239 187.056               68.142               0.491  6866  540.980      0.634         0.800     1867         0.041           343            0.224
       3   A_seen_stations lightgbm_wx 66239 185.450               68.490               0.498  6866  534.859      0.638         0.810     1867         0.070           436            0.298
       3   A_seen_stations        lstm 66239 175.623               66.035               0.507  6866  494.795      0.658         0.809     1867         0.050           265            0.351
       3 B_unseen_stations persistence 39867 183.179               48.378               0.358  3744  533.556      0.717         0.714     1059         0.000             0              NaN
       3 B_unseen_stations    lightgbm 39867 234.617               39.378               0.482  3744  745.587      0.625         0.743     1059         0.098           305            0.341
       3 B_unseen_stations lightgbm_wx 39867 231.833               39.708               0.496  3744  736.130      0.635         0.742     1059         0.139           393            0.374
       3 B_unseen_stations        lstm 39867 200.905               38.186               0.526  3744  633.591      0.641         0.785     1059         0.063           168            0.399
       7   A_seen_stations persistence 68179 261.245               86.614              -0.017  7122  620.881      0.606         0.605     2806         0.000             0              NaN
       7   A_seen_stations    lightgbm 68179 231.782               75.219               0.292  7122  680.574      0.465         0.735     2806         0.062           645            0.268
       7   A_seen_stations lightgbm_wx 68179 223.871               74.233               0.325  7122  654.259      0.472         0.759     2806         0.074           621            0.333
       7   A_seen_stations        lstm 68179 223.496               75.080               0.317  7122  638.278      0.504         0.746     2806         0.073           546            0.377
       7 B_unseen_stations persistence 40942 252.392               64.963              -0.039  3878  712.823      0.600         0.598     1551         0.000             0              NaN
       7 B_unseen_stations    lightgbm 40942 269.730               47.963               0.304  3878  850.819      0.519         0.623     1551         0.139           769            0.281
       7 B_unseen_stations lightgbm_wx 40942 268.635               46.467               0.294  3878  848.069      0.517         0.626     1551         0.153           798            0.297
       7 B_unseen_stations        lstm 40942 252.750               47.217               0.321  3878  795.417      0.503         0.699     1551         0.104           414            0.391
```

## LSTM minus lightgbm_wx (positive = LSTM better)
```
                           nse_lstm_minus_lightgbm_wx  onset_recall_lstm_minus_lightgbm_wx  onset_prec_lstm_minus_lightgbm_wx
horizon split
1       A_seen_stations                         0.006                               -0.048                             -0.039
        B_unseen_stations                       0.032                               -0.059                              0.069
3       A_seen_stations                         0.009                               -0.020                              0.053
        B_unseen_stations                       0.030                               -0.076                              0.025
7       A_seen_stations                        -0.008                               -0.000                              0.044
        B_unseen_stations                       0.028                               -0.048                              0.094
```
