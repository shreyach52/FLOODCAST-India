import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from floodcast.contracts import (DATE, STATION, TARGET, cfs_to_m3s, m3s_to_cfs, make_splits,
                                 validate_canonical_df, validate_forecast_df)


def _synthetic():
    spec = importlib.util.spec_from_file_location(
        "mk", Path(__file__).resolve().parents[1] / "scripts" / "00_make_synthetic.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    df = mod.make(n_stations=10, start="2001-01-01", end="2005-12-31")
    return df.rename(columns={"code": STATION, "q": TARGET})


def test_unit_roundtrip():
    assert cfs_to_m3s(m3s_to_cfs(12.5)) == pytest.approx(12.5)
    assert m3s_to_cfs(1.0) == pytest.approx(35.3147, rel=1e-4)


def test_splits_are_chronological_and_disjoint():
    df = _synthetic()
    train, val, test, ung = make_splits(df)
    assert train[DATE].max() < val[DATE].min()
    assert val[DATE].max() < test[DATE].min()
    seen = set(train[STATION])
    held = set(ung[STATION])
    assert held and not (seen & held), "held-out stations leaked into training"
    assert set(val[STATION]) <= seen and set(test[STATION]) <= seen
    assert ung[DATE].min() >= test[DATE].min()
    assert len(train) + len(val) + len(test) + len(ung) <= len(df)


def test_splits_are_reproducible():
    df = _synthetic()
    a = make_splits(df)[3][STATION].unique()
    b = make_splits(df)[3][STATION].unique()
    assert list(a) == list(b)


def _good_forecast():
    return pd.DataFrame({"station_id": ["A", "A"], "issue_date": ["2024-01-01"] * 2,
                         "horizon": [1, 3], "q_pred": [10.0, 12.0],
                         "q_low": [8.0, 9.0], "q_high": [12.0, 15.0]})


def test_validate_forecast_ok():
    validate_forecast_df(_good_forecast())


@pytest.mark.parametrize("mutate", [
    lambda d: d.assign(horizon=[1, 5]),
    lambda d: d.assign(q_low=[-1.0, 9.0]),
    lambda d: d.assign(q_pred=[7.0, 12.0]),          # pred below low
    lambda d: d.assign(q_high=[12.0, np.nan]),
    lambda d: d.drop(columns=["q_high"]),
])
def test_validate_forecast_rejects_bad(mutate):
    with pytest.raises(AssertionError):
        validate_forecast_df(mutate(_good_forecast()))


def test_validate_canonical():
    df = pd.DataFrame({STATION: ["A", "A"], DATE: pd.to_datetime(["2020-01-01", "2020-01-02"]),
                       TARGET: [1.0, 2.0]})
    validate_canonical_df(df)
    with pytest.raises(AssertionError):
        validate_canonical_df(pd.concat([df, df]))               # duplicates
    with pytest.raises(AssertionError):
        validate_canonical_df(df.assign(**{TARGET: [1.0, -2.0]}))  # negative
