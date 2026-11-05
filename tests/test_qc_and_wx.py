import numpy as np
import pandas as pd

from floodcast.features_wx import WX_FEATURES, weather_features
from floodcast.qc import flag_station


def _series(n=120, base=100.0):
    return pd.Series(base + np.sin(np.arange(n) / 5) * 10, index=pd.date_range("2019-06-01", periods=n))


def test_qc_drops_consecutive_pair_spike():
    s = _series(); s.iloc[50:52] = [150000.0, 160000.0]            # KOLLEGAL-style pair
    f = flag_station(s)
    assert set(f[f.action == "drop"].date) == {s.index[50], s.index[51]}


def test_qc_drops_spike_across_missing_day():
    s = _series(); s.iloc[50:53] = [120000.0, 110000.0, 99000.0]; s = s.drop(s.index[51])
    f = flag_station(s)
    assert (f.action == "drop").sum() == 2


def test_qc_keeps_real_multiday_flood():
    s = _series(); s.iloc[50:60] = [300, 900, 2500, 6000, 12000, 15000, 11000, 6000, 2500, 900]
    assert (flag_station(s).action == "drop").sum() == 0


def test_qc_abs_cap():
    s = _series(); s.iloc[10:20] = 80000.0                          # long run, still impossible
    assert (flag_station(s).rule == "C_above_abs_cap").sum() == 10


def _wx(n=100):
    d = pd.date_range("2020-01-01", periods=n)
    return pd.DataFrame({"station_id": "A", "date": d, "rain_mm": np.arange(n, dtype=float) % 7, "soil1": 0.2})


def test_wx_no_future_leakage():
    w = _wx(); a = weather_features(w, "2020-03-01")
    w2 = w.copy(); w2.loc[w2.date >= "2020-02-10", "rain_mm"] = 999.0   # change the future
    b = weather_features(w2, "2020-03-01")
    cut = pd.Timestamp("2020-02-09")
    ca, cb = a[a.date <= cut][list(WX_FEATURES)].drop(columns=["rain7_rel", "rain30_rel"]), \
             b[b.date <= cut][list(WX_FEATURES)].drop(columns=["rain7_rel", "rain30_rel"])
    pd.testing.assert_frame_equal(ca.reset_index(drop=True), cb.reset_index(drop=True))


def test_wx_rolling_sum_and_columns():
    f = weather_features(_wx(), "2020-03-01").set_index("date")
    r = _wx().set_index("date").rain_mm
    assert abs(f.loc["2020-02-01", "rain7"] - r.loc["2020-01-26":"2020-02-01"].sum()) < 1e-9
    assert list(f.columns) == ["station_id"] + list(WX_FEATURES)
def test_wx_one_day_lag_uses_previous_day_features():
    w = _wx()

    current = weather_features(
        w, "2020-03-01", lag_days=0
    ).set_index("date")

    lagged = weather_features(
        w, "2020-03-01", lag_days=1
    ).set_index("date")

    target_date = pd.Timestamp("2020-02-15")
    previous_date = target_date - pd.Timedelta(days=1)

    np.testing.assert_allclose(
        lagged.loc[target_date, list(WX_FEATURES)].to_numpy(dtype=float),
        current.loc[previous_date, list(WX_FEATURES)].to_numpy(dtype=float),
        equal_nan=True
    )