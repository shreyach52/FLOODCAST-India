# Value of future rain (diagnostic, NOT a deployable model)

Train <= 2018-12-31, val <= 2021-12-31, test >= 2022-01-01. Quick mode: True.
Stations: 20 seen (split A), 10 unseen (split B).

`wx` = discharge + past weather only. `fc_rhoX` = also a rain forecast for days t+1..t+h whose correlation with the rain that really fell is X. `oracle` = perfect knowledge (X=1). All variants use the same model, features and splits; only the future-rain features differ. Forecast errors here are random noise of the stated correlation: real forecasts also have biases and miss extremes, so treat the curve as optimistic.

## Threshold-free early warning (onset = below threshold today, at/above it h days later)
```
split  horizon       model  n_rows  n_onsets  prevalence  roc_auc  pr_auc  tpr@0.01  tpr@0.02  tpr@0.05
    A        1 persistence   17263       377      0.0218   0.8566  0.2277    0.2467    0.3660    0.5782
    A        1          wx   17263       377      0.0218   0.9132  0.1985    0.1883    0.3554    0.5650
    A        1   fc_rho0.5   17263       377      0.0218   0.9106  0.1818    0.1194    0.3422    0.5623
    A        1   fc_rho0.7   17263       377      0.0218   0.9125  0.1823    0.1167    0.3554    0.5597
    A        1   fc_rho0.9   17263       377      0.0218   0.9128  0.1731    0.0822    0.2997    0.5703
    A        1      oracle   17263       377      0.0218   0.9151  0.1714    0.1061    0.2626    0.5756
    B        1 persistence   10574       225      0.0213   0.8652  0.1871    0.2044    0.2889    0.4578
    B        1          wx   10574       225      0.0213   0.8869  0.1858    0.2000    0.2667    0.4978
    B        1   fc_rho0.5   10574       225      0.0213   0.8904  0.1945    0.2000    0.2933    0.4889
    B        1   fc_rho0.7   10574       225      0.0213   0.8916  0.1916    0.1911    0.2800    0.4889
    B        1   fc_rho0.9   10574       225      0.0213   0.8975  0.1930    0.1689    0.2844    0.5200
    B        1      oracle   10574       225      0.0213   0.9016  0.1875    0.1778    0.2978    0.5111
    A        3 persistence   16992       617      0.0363   0.8303  0.2002    0.1280    0.2350    0.4198
    A        3          wx   16992       617      0.0363   0.8634  0.1607    0.0470    0.1507    0.3501
    A        3   fc_rho0.5   16992       617      0.0363   0.8693  0.1634    0.0389    0.1264    0.3679
    A        3   fc_rho0.7   16992       617      0.0363   0.8725  0.1650    0.0438    0.1378    0.3566
    A        3   fc_rho0.9   16992       617      0.0363   0.8789  0.1685    0.0519    0.1280    0.3566
    A        3      oracle   16992       617      0.0363   0.8805  0.1741    0.0486    0.1297    0.3647
    B        3 persistence   10471       395      0.0377   0.8298  0.1743    0.1241    0.1747    0.3165
    B        3          wx   10471       395      0.0377   0.8538  0.2002    0.1089    0.2101    0.3595
    B        3   fc_rho0.5   10471       395      0.0377   0.8617  0.2127    0.1215    0.2304    0.3899
    B        3   fc_rho0.7   10471       395      0.0377   0.8734  0.2279    0.1114    0.2380    0.4203
    B        3   fc_rho0.9   10471       395      0.0377   0.8860  0.2398    0.1266    0.2734    0.4430
    B        3      oracle   10471       395      0.0377   0.8927  0.2421    0.1165    0.2329    0.4810
    A        7 persistence   17576       939      0.0534   0.8170  0.2129    0.0831    0.1619    0.3067
    A        7          wx   17576       939      0.0534   0.8353  0.1872    0.0437    0.0703    0.2812
    A        7   fc_rho0.5   17576       939      0.0534   0.8399  0.1928    0.0405    0.0937    0.2758
    A        7   fc_rho0.7   17576       939      0.0534   0.8473  0.2028    0.0426    0.0927    0.3078
    A        7   fc_rho0.9   17576       939      0.0534   0.8571  0.2065    0.0426    0.1033    0.2875
    A        7      oracle   17576       939      0.0534   0.8650  0.2079    0.0405    0.0852    0.2588
    B        7 persistence   10702       542      0.0506   0.7915  0.1626    0.0812    0.1218    0.2343
    B        7          wx   10702       542      0.0506   0.8246  0.2146    0.1089    0.1624    0.2952
    B        7   fc_rho0.5   10702       542      0.0506   0.8390  0.2273    0.1125    0.1679    0.3081
    B        7   fc_rho0.7   10702       542      0.0506   0.8506  0.2376    0.0996    0.1863    0.3229
    B        7   fc_rho0.9   10702       542      0.0506   0.8743  0.2582    0.0996    0.1919    0.3782
    B        7      oracle   10702       542      0.0506   0.8905  0.2815    0.1033    0.1808    0.4446
```

## Gain over `wx`, with station-block 95% interval (A minus B)
```
 horizon split         A  B   metric    diff      lo      hi  prob_a_better  n_boot
       1     A fc_rho0.5 wx  roc_auc -0.0026 -0.0065 -0.0001           0.05      20
       1     A fc_rho0.5 wx tpr@0.02 -0.0133 -0.1357  0.0189           0.10      20
       1     A fc_rho0.7 wx  roc_auc -0.0007 -0.0054  0.0027           0.40      20
       1     A fc_rho0.7 wx tpr@0.02  0.0000 -0.1338  0.0337           0.30      20
       1     A fc_rho0.9 wx  roc_auc -0.0004 -0.0062  0.0040           0.45      20
       1     A fc_rho0.9 wx tpr@0.02 -0.0557 -0.1649  0.0127           0.15      20
       1     A    oracle wx  roc_auc  0.0019 -0.0057  0.0082           0.70      20
       1     A    oracle wx tpr@0.02 -0.0928 -0.1676  0.0149           0.15      20
       1     B fc_rho0.5 wx  roc_auc  0.0036  0.0007  0.0061           1.00      20
       1     B fc_rho0.5 wx tpr@0.02  0.0267 -0.0223  0.0854           0.70      20
       1     B fc_rho0.7 wx  roc_auc  0.0048 -0.0003  0.0074           0.95      20
       1     B fc_rho0.7 wx tpr@0.02  0.0133 -0.0508  0.0745           0.60      20
       1     B fc_rho0.9 wx  roc_auc  0.0107  0.0050  0.0145           1.00      20
       1     B fc_rho0.9 wx tpr@0.02  0.0178 -0.0364  0.1054           0.70      20
       1     B    oracle wx  roc_auc  0.0148  0.0058  0.0196           1.00      20
       1     B    oracle wx tpr@0.02  0.0311 -0.0733  0.1434           0.65      20
       3     A fc_rho0.5 wx  roc_auc  0.0059  0.0016  0.0073           0.95      20
       3     A fc_rho0.5 wx tpr@0.02 -0.0243 -0.0481  0.0256           0.15      20
       3     A fc_rho0.7 wx  roc_auc  0.0091  0.0028  0.0123           1.00      20
       3     A fc_rho0.7 wx tpr@0.02 -0.0130 -0.0419  0.0336           0.25      20
       3     A fc_rho0.9 wx  roc_auc  0.0155  0.0049  0.0222           1.00      20
       3     A fc_rho0.9 wx tpr@0.02 -0.0227 -0.0505  0.0277           0.40      20
       3     A    oracle wx  roc_auc  0.0171  0.0051  0.0253           1.00      20
       3     A    oracle wx tpr@0.02 -0.0211 -0.0641  0.0594           0.45      20
       3     B fc_rho0.5 wx  roc_auc  0.0078  0.0027  0.0118           1.00      20
       3     B fc_rho0.5 wx tpr@0.02  0.0203 -0.0033  0.0554           0.90      20
       3     B fc_rho0.7 wx  roc_auc  0.0196  0.0111  0.0253           1.00      20
       3     B fc_rho0.7 wx tpr@0.02  0.0278 -0.0212  0.0831           0.85      20
       3     B fc_rho0.9 wx  roc_auc  0.0321  0.0167  0.0402           1.00      20
       3     B fc_rho0.9 wx tpr@0.02  0.0633 -0.0048  0.1480           0.90      20
       3     B    oracle wx  roc_auc  0.0388  0.0240  0.0454           1.00      20
       3     B    oracle wx tpr@0.02  0.0228 -0.0155  0.2295           0.70      20
       7     A fc_rho0.5 wx  roc_auc  0.0046  0.0015  0.0096           1.00      20
       7     A fc_rho0.5 wx tpr@0.02  0.0234 -0.0296  0.0322           0.45      20
       7     A fc_rho0.7 wx  roc_auc  0.0120  0.0058  0.0219           1.00      20
       7     A fc_rho0.7 wx tpr@0.02  0.0224 -0.0335  0.0497           0.65      20
       7     A fc_rho0.9 wx  roc_auc  0.0218  0.0103  0.0346           1.00      20
       7     A fc_rho0.9 wx tpr@0.02  0.0330 -0.0495  0.0570           0.50      20
       7     A    oracle wx  roc_auc  0.0297  0.0115  0.0461           1.00      20
       7     A    oracle wx tpr@0.02  0.0149 -0.0946  0.0638           0.35      20
       7     B fc_rho0.5 wx  roc_auc  0.0145  0.0053  0.0212           1.00      20
       7     B fc_rho0.5 wx tpr@0.02  0.0055 -0.0131  0.0289           0.70      20
       7     B fc_rho0.7 wx  roc_auc  0.0261  0.0140  0.0338           1.00      20
       7     B fc_rho0.7 wx tpr@0.02  0.0240 -0.0166  0.0654           0.80      20
       7     B fc_rho0.9 wx  roc_auc  0.0497  0.0323  0.0602           1.00      20
       7     B fc_rho0.9 wx tpr@0.02  0.0295 -0.0166  0.1235           0.85      20
       7     B    oracle wx  roc_auc  0.0659  0.0466  0.0765           1.00      20
       7     B    oracle wx tpr@0.02  0.0185 -0.0183  0.2158           0.75      20
```

## Classic metrics at the fixed alarm rule (for continuity with the main tables)
```
 horizon split       model  median_station_nse  onset_recall  onset_precision
       1     A persistence               0.695         0.000              NaN
       1     A          wx               0.709         0.159            0.276
       1     A   fc_rho0.5               0.709         0.172            0.243
       1     A   fc_rho0.7               0.715         0.191            0.247
       1     A   fc_rho0.9               0.706         0.207            0.220
       1     A      oracle               0.672         0.210            0.206
       1     B persistence               0.652         0.000              NaN
       1     B          wx               0.684         0.156            0.343
       1     B   fc_rho0.5               0.683         0.178            0.345
       1     B   fc_rho0.7               0.690         0.178            0.305
       1     B   fc_rho0.9               0.700         0.178            0.268
       1     B      oracle               0.728         0.213            0.262
       3     A persistence               0.347         0.000              NaN
       3     A          wx               0.450         0.146            0.217
       3     A   fc_rho0.5               0.453         0.169            0.205
       3     A   fc_rho0.7               0.454         0.203            0.212
       3     A   fc_rho0.9               0.417         0.259            0.214
       3     A      oracle               0.464         0.316            0.221
       3     B persistence               0.403         0.000              NaN
       3     B          wx               0.546         0.165            0.307
       3     B   fc_rho0.5               0.541         0.185            0.330
       3     B   fc_rho0.7               0.561         0.218            0.316
       3     B   fc_rho0.9               0.588         0.284            0.352
       3     B      oracle               0.619         0.278            0.319
       7     A persistence              -0.035         0.000              NaN
       7     A          wx               0.241         0.160            0.225
       7     A   fc_rho0.5               0.227         0.181            0.225
       7     A   fc_rho0.7               0.222         0.216            0.243
       7     A   fc_rho0.9               0.259         0.278            0.241
       7     A      oracle               0.324         0.359            0.246
       7     B persistence               0.059         0.000              NaN
       7     B          wx               0.328         0.188            0.291
       7     B   fc_rho0.5               0.329         0.208            0.288
       7     B   fc_rho0.7               0.334         0.255            0.306
       7     B   fc_rho0.9               0.338         0.256            0.296
       7     B      oracle               0.374         0.336            0.340
```
