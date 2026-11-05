import numpy as np
import pandas as pd
import pytest

from floodcast.features_wx import WX_FEATURES, weather_features
from floodcast.futurerain import (DIS_FEATURES, ORACLE_PREFIX, degrade, discharge_frame, future_rain,
                                  oracle_cols, zscore_log)


def _w(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"station_id": "A", "date": pd.date_range("2020-01-01", periods=n),
                         "rain_mm": rng.gamma(0.6, 6, n), "soil1": 0.2 + 0.01 * rng.random(n)})


def test_future_rain_uses_exactly_days_t_plus_1_to_t_plus_h():
    w = _w()
    f = future_rain(w).set_index("date")
    r = w.set_index("date")["rain_mm"]
    t = pd.Timestamp("2020-01-10")
    assert f.loc[t, "oracle_sum3"] == pytest.approx(r.loc["2020-01-11":"2020-01-13"].sum())
    assert f.loc[t, "oracle_max7"] == pytest.approx(r.loc["2020-01-11":"2020-01-17"].max())
    assert f.loc[t, "oracle_sum1"] == pytest.approx(r.loc["2020-01-11"])
    # changing rain on day t itself or on t+h+1 must NOT change the t features
    w2 = w.copy()
    w2.loc[w2.date == "2020-01-10", "rain_mm"] += 100
    w2.loc[w2.date == "2020-01-14", "rain_mm"] += 100
    f2 = future_rain(w2).set_index("date")
    assert f2.loc[t, "oracle_sum3"] == pytest.approx(f.loc[t, "oracle_sum3"])
    # but changing day t+1 must change it
    w3 = w.copy()
    w3.loc[w3.date == "2020-01-11", "rain_mm"] += 100
    assert future_rain(w3).set_index("date").loc[t, "oracle_sum3"] > f.loc[t, "oracle_sum3"] + 50


def test_last_days_have_no_future_and_gaps_give_nan():
    w = _w()
    f = future_rain(w).set_index("date")
    assert f["oracle_sum7"].iloc[-7:].isna().all() and f["oracle_sum7"].iloc[-8] == f["oracle_sum7"].iloc[-8]
    w = w.drop(index=20)                                   # one missing day
    f = future_rain(w).set_index("date")
    assert f["oracle_sum3"].isna().sum() >= 3 + 3          # windows touching the gap + the tail


def test_normal_weather_features_never_see_tomorrow():
    w = _w()
    base = weather_features(w, "2020-02-01").set_index("date")
    w2 = w.copy()
    w2.loc[w2.date == "2020-01-20", "rain_mm"] += 500       # a future event relative to 2020-01-19
    after = weather_features(w2, "2020-02-01").set_index("date")
    cols = [c for c in WX_FEATURES if c not in ("rain7_rel", "rain30_rel")]   # *_rel use train mean scale
    pd.testing.assert_frame_equal(base.loc[:"2020-01-19", cols], after.loc[:"2020-01-19", cols])


def test_oracle_columns_are_named_and_isolated():
    assert all(c.startswith(ORACLE_PREFIX) for h in (1, 3, 7) for c in oracle_cols(h))
    assert oracle_cols(1) == ["oracle_sum1"] and oracle_cols(7) == ["oracle_sum7", "oracle_max7"]
    assert not any(c.startswith(ORACLE_PREFIX) for c in DIS_FEATURES + list(WX_FEATURES))


def test_degrade_has_requested_correlation_and_nests():
    rng = np.random.default_rng(0)
    z, eps = rng.standard_normal(50000), rng.standard_normal(50000)
    for rho in (0.3, 0.7, 0.9):
        assert np.corrcoef(degrade(z, rho, eps), z)[0, 1] == pytest.approx(rho, abs=0.02)
    assert np.array_equal(degrade(z, 1.0, eps), z)
    assert abs(np.corrcoef(degrade(z, 0.0, eps), z)[0, 1]) < 0.02
    with pytest.raises(ValueError):
        degrade(z, 1.2, eps)
    nan = np.array([np.nan, 1.0])
    assert np.isnan(degrade(nan, 0.5, np.zeros(2))[0])


def test_zscore_uses_only_reference_period():
    d = pd.DataFrame({"station_id": "A", "date": pd.date_range("2020-01-01", periods=10),
                      "x": [1., 2, 3, 4, 5, 1000, 1000, 1000, 1000, 1000]})
    z = zscore_log(d, ["x"], {"A": pd.Timestamp("2020-01-05")})
    assert z["x"].iloc[:5].mean() == pytest.approx(0, abs=1e-9)      # standardised on first 5 days
    assert z["x"].iloc[5] > 10                                       # later huge values are not 'absorbed'


def test_discharge_frame_targets_and_no_leak():
    g = pd.DataFrame({"station_id": "A", "date": pd.date_range("2020-01-01", periods=40),
                      "discharge": np.arange(1.0, 41.0)})
    f = discharge_frame(g).set_index("date")
    t = pd.Timestamp("2020-01-20")
    assert f.loc[t, "y3"] == 23.0 and f.loc[t, "q0"] == 20.0
    assert f.loc[t, "lag1"] == pytest.approx(np.log1p(19.0))
    g2 = g.copy()
    g2.loc[g2.date == "2020-01-21", "discharge"] = 9999.0           # tomorrow changes...
    f2 = discharge_frame(g2).set_index("date")
    assert f2.loc[t, DIS_FEATURES].equals(f.loc[t, DIS_FEATURES])   # ...today's features do not
