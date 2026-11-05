# Baseline report (discharge + ERA5-Land weather)

Train <= 2018-12-31, val <= 2021-12-31, test >= 2022-01-01. Stations used: 98. Unseen stations: 10 random held-out ['BANTWAL', 'BELUR', 'ERINJIPUZHA', 'KOKKEDODDY', 'MUTHANKERA', 'Nagalamedike', 'Navalgund', 'Saigaon', 'THORESHETTAHALLI', 'Yadgir'] + 25 late-start (record begins after train end, no training data) ['ADKASTHALA', 'ARKKANNUR', 'ARUVIPURAM', 'Airani', 'CHAKKALAKUTH', 'HARIHARAPURA', 'HOMMARAGALLI', 'JANWADA', 'KONDAZHY', 'KOTTAPURAM', 'Kalmane', 'LAKKAVALLI', 'MADAMON', 'MAHISHI', 'MANAKKAD', 'MANIKAL', 'NELLIPPALLI', 'PALAKADAVU', 'PALLIPPADY', 'PERUVAMPADAM', 'POOKKAYAM', 'PULIKUKKU', 'PULLAKKAYAR', 'THOTTATHINKADAVU', 'VELLAIKADAVU']

High flow = station's train-period 95th percentile. hi_recall = share of true high-flow days whose forecast also reached that level. onset_* = only days where the river is BELOW its threshold today (persistence scores 0 by construction): onset_recall = share of true crossings flagged in advance, onset_precision = share of alarms that were real. median_station_rmse / median_station_nse are robust to a few flashy stations; judge progress with them and onset_*.

```
 horizon             split       model     n    rmse    mae    r2  hi_rmse  median_station_rmse  median_station_nse  hi_n  hi_recall  hi_precision  onset_n  onset_recall  onset_alarms  onset_precision
       1   A_seen_stations persistence 67056 116.343 24.890 0.838  298.597               48.310               0.720  6965      0.843         0.842     1094         0.000             0              NaN
       1   A_seen_stations climatology 67056 255.401 56.005 0.221  755.657               79.402               0.193  6965      0.221         0.907     1094         0.005            87            0.057
       1   A_seen_stations    lightgbm 67056 115.955 24.460 0.840  317.700               46.354               0.751  6965      0.796         0.879     1094         0.053           175            0.331
       1   A_seen_stations lightgbm_wx 67056 116.006 24.007 0.839  320.458               45.571               0.741  6965      0.801         0.876     1094         0.088           254            0.378
       1 B_unseen_stations persistence 40269 106.769 20.484 0.886  314.398               37.015               0.742  3814      0.845         0.841      592         0.000             0              NaN
       1 B_unseen_stations climatology 40269 289.133 56.227 0.161  883.781               61.591               0.017  3814      0.396         0.712      592         0.030           490            0.037
       1 B_unseen_stations    lightgbm 40269 144.035 24.188 0.792  449.293               35.666               0.721  3814      0.799         0.858      592         0.071           112            0.375
       1 B_unseen_stations lightgbm_wx 40269 129.065 22.683 0.833  399.814               31.380               0.735  3814      0.811         0.846      592         0.111           195            0.338
       3   A_seen_stations persistence 66239 189.526 43.488 0.575  482.587               81.537               0.375  6866      0.728         0.725     1867         0.000             0              NaN
       3   A_seen_stations climatology 66239 256.549 56.508 0.221  759.967               79.955               0.172  6866      0.210         0.873     1867         0.007            90            0.144
       3   A_seen_stations    lightgbm 66239 187.056 38.826 0.586  540.980               68.142               0.491  6866      0.634         0.800     1867         0.041           343            0.224
       3   A_seen_stations lightgbm_wx 66239 185.450 37.931 0.593  534.859               68.490               0.498  6866      0.638         0.810     1867         0.070           436            0.298
       3 B_unseen_stations persistence 39867 183.179 36.794 0.664  533.556               48.378               0.358  3744      0.717         0.714     1059         0.000             0              NaN
       3 B_unseen_stations climatology 39867 289.620 56.485 0.161  889.413               62.485               0.021  3744      0.373         0.670     1059         0.035           464            0.080
       3 B_unseen_stations    lightgbm 39867 234.617 38.931 0.449  745.587               39.378               0.482  3744      0.625         0.743     1059         0.098           305            0.341
       3 B_unseen_stations lightgbm_wx 39867 231.833 38.578 0.462  736.130               39.708               0.496  3744      0.635         0.742     1059         0.139           393            0.374
       7   A_seen_stations persistence 68179 261.245 60.591 0.196  620.881               86.614              -0.017  7122      0.606         0.605     2806         0.000             0              NaN
       7   A_seen_stations climatology 68179 256.888 57.301 0.223  758.344               79.360               0.095  7122      0.195         0.823     2806         0.007            89            0.236
       7   A_seen_stations    lightgbm 68179 231.782 48.763 0.367  680.574               75.219               0.292  7122      0.465         0.735     2806         0.062           645            0.268
       7   A_seen_stations lightgbm_wx 68179 223.871 47.748 0.410  654.259               74.233               0.325  7122      0.472         0.759     2806         0.074           621            0.333
       7 B_unseen_stations persistence 40942 252.392 51.604 0.358  712.823               64.963              -0.039  3878      0.600         0.598     1551         0.000             0              NaN
       7 B_unseen_stations climatology 40942 285.222 56.809 0.180  880.728               65.963              -0.005  3878      0.349         0.636     1551         0.050           477            0.164
       7 B_unseen_stations    lightgbm 40942 269.730 48.732 0.267  850.819               47.963               0.304  3878      0.519         0.623     1551         0.139           769            0.281
       7 B_unseen_stations lightgbm_wx 40942 268.635 48.255 0.273  848.069               46.467               0.294  3878      0.517         0.626     1551         0.153           798            0.297
```

## What the weather adds (lightgbm_wx minus lightgbm; negative RMSE change = better)
```
                           med_rmse_change  nse_change  onset_recall_wo  onset_recall_wx  onset_prec_wo  onset_prec_wx
horizon split                                                                                                         
1       A_seen_stations             -0.783      -0.010            0.053            0.088          0.331          0.378
        B_unseen_stations           -4.286       0.015            0.071            0.111          0.375          0.338
3       A_seen_stations              0.348       0.007            0.041            0.070          0.224          0.298
        B_unseen_stations            0.330       0.014            0.098            0.139          0.341          0.374
7       A_seen_stations             -0.986       0.032            0.062            0.074          0.268          0.333
        B_unseen_stations           -1.497      -0.010            0.139            0.153          0.281          0.297
```

## Feature importance (weather model)
```
h=1d top features (gain share): diff1 18%, q_rel_med 6%, rain1 6%, ratio7 6%, q0 6%, lon 5%, diff3 4%, lag2 4%
h=3d top features (gain share): q_rel_med 10%, diff1 7%, doy_cos 7%, rain1 6%, diff3 6%, doy_sin 6%, q0 5%, ratio7 5%
h=7d top features (gain share): q_rel_med 14%, doy_cos 12%, doy_sin 8%, q0 7%, lon 6%, lat 5%, log_station_med 5%, diff3 5%
```