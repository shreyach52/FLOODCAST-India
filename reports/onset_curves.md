# Threshold-free early-warning comparison

Onset = river below its high-flow threshold today and at/above it h days later. Score = predicted discharge / threshold. `persistence` is q0/threshold (a real score here).

- **roc_auc**: 0.5 = coin flip, 1 = perfect ranking.
- **pr_auc**: compare with `prevalence` (the score of a random guess).
- **tpr@F**: share of onsets caught while only F of non-onset days may raise a false alarm.
- Differences are only meaningful when the 95% interval (lo, hi) excludes 0.

## Per model

```
            split  horizon       model  n_rows  n_onsets  prevalence  roc_auc  pr_auc  tpr@0.01  tpr@0.02  tpr@0.05
  A_seen_stations        1 persistence   52902       930      0.0176   0.8643  0.1865    0.2258    0.3613    0.5581
  A_seen_stations        1    LightGBM   52902       930      0.0176   0.9103  0.1901    0.2161    0.3688    0.5753
  A_seen_stations        1        LSTM   52902       930      0.0176   0.9126  0.1973    0.2538    0.3763    0.5785
B_unseen_stations        1 persistence    5675        96      0.0169   0.8568  0.1837    0.2188    0.3229    0.5312
B_unseen_stations        1    LightGBM    5675        96      0.0169   0.9138  0.1869    0.1979    0.3229    0.5104
B_unseen_stations        1        LSTM    5675        96      0.0169   0.9096  0.1862    0.2396    0.3438    0.5417
  A_seen_stations        3 persistence   52264      1617      0.0309   0.8425  0.1752    0.1150    0.2165    0.4007
  A_seen_stations        3    LightGBM   52264      1617      0.0309   0.8756  0.1746    0.1107    0.2022    0.4094
  A_seen_stations        3        LSTM   52264      1617      0.0309   0.8798  0.2060    0.1447    0.2443    0.4576
B_unseen_stations        3 persistence    5592       163      0.0291   0.8579  0.2108    0.1595    0.2209    0.4233
B_unseen_stations        3    LightGBM    5592       163      0.0291   0.8900  0.2164    0.1718    0.2761    0.4785
B_unseen_stations        3        LSTM    5592       163      0.0291   0.8908  0.2342    0.2086    0.2883    0.4724
  A_seen_stations        7 persistence   53827      2445      0.0454   0.8178  0.1753    0.0728    0.1415    0.2871
  A_seen_stations        7    LightGBM   53827      2445      0.0454   0.8508  0.1963    0.0703    0.1501    0.3292
  A_seen_stations        7        LSTM   53827      2445      0.0454   0.8527  0.2265    0.1174    0.1845    0.3603
B_unseen_stations        7 persistence    5770       245      0.0425   0.8095  0.1645    0.0612    0.1224    0.2735
B_unseen_stations        7    LightGBM    5770       245      0.0425   0.8465  0.2021    0.0980    0.1878    0.3429
B_unseen_stations        7        LSTM    5770       245      0.0425   0.8450  0.2470    0.1429    0.2327    0.3673
```

## Paired differences (A minus B, station-block bootstrap)

```
 horizon             split        A           B   metric    diff      lo      hi  prob_a_better  n_boot
       1   A_seen_stations LightGBM        LSTM  roc_auc -0.0024 -0.0052  0.0004          0.050     300
       1   A_seen_stations LightGBM        LSTM tpr@0.02 -0.0075 -0.0518  0.0257          0.350     300
       1   A_seen_stations LightGBM persistence  roc_auc  0.0460  0.0139  0.0820          1.000     300
       1   A_seen_stations LightGBM persistence tpr@0.02  0.0075 -0.0637  0.0562          0.677     300
       1   A_seen_stations     LSTM persistence  roc_auc  0.0484  0.0156  0.0845          1.000     300
       1   A_seen_stations     LSTM persistence tpr@0.02  0.0151 -0.0154  0.0427          0.830     300
       1 B_unseen_stations LightGBM        LSTM  roc_auc  0.0043 -0.0027  0.0118          0.910     300
       1 B_unseen_stations LightGBM        LSTM tpr@0.02 -0.0208 -0.0514  0.0402          0.327     300
       1 B_unseen_stations LightGBM persistence  roc_auc  0.0570  0.0063  0.1386          1.000     300
       1 B_unseen_stations LightGBM persistence tpr@0.02  0.0000 -0.0513  0.0457          0.437     300
       1 B_unseen_stations     LSTM persistence  roc_auc  0.0527  0.0030  0.1315          1.000     300
       1 B_unseen_stations     LSTM persistence tpr@0.02  0.0208 -0.0476  0.0725          0.510     300
       3   A_seen_stations LightGBM        LSTM  roc_auc -0.0041 -0.0083 -0.0007          0.007     300
       3   A_seen_stations LightGBM        LSTM tpr@0.02 -0.0421 -0.0740 -0.0094          0.007     300
       3   A_seen_stations LightGBM persistence  roc_auc  0.0331  0.0130  0.0545          1.000     300
       3   A_seen_stations LightGBM persistence tpr@0.02 -0.0142 -0.0829  0.0471          0.383     300
       3   A_seen_stations     LSTM persistence  roc_auc  0.0373  0.0165  0.0600          1.000     300
       3   A_seen_stations     LSTM persistence tpr@0.02  0.0278 -0.0298  0.0819          0.837     300
       3 B_unseen_stations LightGBM        LSTM  roc_auc -0.0007 -0.0093  0.0088          0.370     300
       3 B_unseen_stations LightGBM        LSTM tpr@0.02 -0.0123 -0.0833  0.0438          0.130     300
       3 B_unseen_stations LightGBM persistence  roc_auc  0.0321  0.0052  0.0564          0.997     300
       3 B_unseen_stations LightGBM persistence tpr@0.02  0.0552 -0.0464  0.1390          0.793     300
       3 B_unseen_stations     LSTM persistence  roc_auc  0.0328  0.0096  0.0564          1.000     300
       3 B_unseen_stations     LSTM persistence tpr@0.02  0.0675 -0.0145  0.1744          0.933     300
       7   A_seen_stations LightGBM        LSTM  roc_auc -0.0019 -0.0097  0.0054          0.343     300
       7   A_seen_stations LightGBM        LSTM tpr@0.02 -0.0344 -0.0766  0.0181          0.103     300
       7   A_seen_stations LightGBM persistence  roc_auc  0.0330  0.0229  0.0419          1.000     300
       7   A_seen_stations LightGBM persistence tpr@0.02  0.0086 -0.0641  0.0783          0.597     300
       7   A_seen_stations     LSTM persistence  roc_auc  0.0348  0.0270  0.0432          1.000     300
       7   A_seen_stations     LSTM persistence tpr@0.02  0.0429 -0.0048  0.0880          0.970     300
       7 B_unseen_stations LightGBM        LSTM  roc_auc  0.0015 -0.0162  0.0279          0.567     300
       7 B_unseen_stations LightGBM        LSTM tpr@0.02 -0.0449 -0.1375  0.0744          0.183     300
       7 B_unseen_stations LightGBM persistence  roc_auc  0.0370  0.0137  0.0695          0.997     300
       7 B_unseen_stations LightGBM persistence tpr@0.02  0.0653 -0.0346  0.1720          0.767     300
       7 B_unseen_stations     LSTM persistence  roc_auc  0.0354  0.0202  0.0567          1.000     300
       7 B_unseen_stations     LSTM persistence tpr@0.02  0.1102  0.0536  0.1621          1.000     300
```
