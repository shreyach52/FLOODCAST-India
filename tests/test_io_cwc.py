import numpy as np
import pandas as pd
import pytest

from floodcast.contracts import DATE, STATION, TARGET, validate_canonical_df
from floodcast.io_cwc import (canonicalize, eligible_stations, guess_column_map, parse_dates,
                              station_coverage)


def test_guess_two_header_styles():
    a = guess_column_map(["Station Code", "Station Name", "River", "State", "Latitude",
                          "Longitude", "Date", "Discharge (cumecs)"])
    assert a[STATION] == "Station Code" and a["station_name"] == "Station Name"
    assert a[DATE] == "Date" and a[TARGET] == "Discharge (cumecs)"
    assert a["lat"] == "Latitude" and a["lon"] == "Longitude"
    b = guess_column_map(["STATION_ID", "STATION_NAME", "RIVER_NAME", "OBS_DATE", "DISCHARGE"])
    assert b[STATION] == "STATION_ID" and b["station_name"] == "STATION_NAME"
    assert b[DATE] == "OBS_DATE" and b[TARGET] == "DISCHARGE"


def test_guess_never_reuses_a_column():
    m = guess_column_map(["Station", "Date", "Discharge"])
    assert len(set(m.values())) == len(m)
    assert m[STATION] == "Station"


def _d(vals):
    return parse_dates(pd.Series(vals))


def test_dates_iso():
    out, how = _d(["2001-06-05", "2001-06-06"])
    assert out.tolist() == [pd.Timestamp("2001-06-05"), pd.Timestamp("2001-06-06")]
    assert "ISO" in how


def test_dates_dayfirst_detected_not_swapped():
    # 02-01-2006 must be 2 Jan, not 1 Feb. The 13th proves day-first.
    out, how = _d(["01-01-2006", "02-01-2006", "13-01-2006"])
    assert out.tolist() == [pd.Timestamp("2006-01-01"), pd.Timestamp("2006-01-02"),
                            pd.Timestamp("2006-01-13")]
    assert how == "day-first (detected)"


def test_dates_monthfirst_detected():
    out, how = _d(["01/02/2006", "01/13/2006"])
    assert out.tolist() == [pd.Timestamp("2006-01-02"), pd.Timestamp("2006-01-13")]
    assert how == "month-first (detected)"


def test_dates_all_ambiguous_flagged():
    out, how = _d(["01-02-2006", "03-04-2006"])
    assert out[0] == pd.Timestamp("2006-02-01")        # day-first assumed
    assert "ASSUMED" in how


def test_dates_text_months_and_garbage():
    out, how = _d(["05-Jun-2001", "not a date"])
    assert out[0] == pd.Timestamp("2001-06-05") and pd.isna(out[1])


def test_dates_real_file_style_regression():
    # regression for the bug where pandas guessed per row and swapped day/month
    days = pd.date_range("2006-01-01", "2006-12-31")
    s = pd.Series(days.strftime("%d-%m-%Y"))
    out, _ = parse_dates(s)
    assert (out.values == days.values).all()


def _raw():
    return pd.DataFrame({
        "Station Code": ["A", "A", "A", "B", "B", " C "],
        "Date": ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-01", "bad", "2020-01-03"],
        "Discharge": ["10", "20", "-999", "5", "7", "abc"],
        "Latitude": ["12.5"] * 6,
    })


def test_canonicalize_cleaning():
    df, st = canonicalize(_raw())
    validate_canonical_df(df)
    a1 = df[(df[STATION] == "A") & (df[DATE] == "2020-01-01")][TARGET].iloc[0]
    assert a1 == 15.0                                       # duplicates averaged
    assert st["bad_dates"] == 1 and st["negative_discharge"] == 1 and st["bad_discharge"] == 1
    assert st["duplicates_merged"] == 1
    assert "C" not in set(df[STATION])                       # its only value was non-numeric
    assert (df[TARGET] >= 0).all()


def test_canonicalize_cfs_conversion():
    raw = pd.DataFrame({"station": ["A"], "date": ["2020-01-01"], "discharge": ["353.146667"]})
    df, _ = canonicalize(raw, source_unit="cfs")
    assert df[TARGET].iloc[0] == pytest.approx(10.0, rel=1e-5)


def test_canonicalize_missing_required_raises():
    with pytest.raises(ValueError):
        canonicalize(pd.DataFrame({"foo": ["1"], "bar": ["2"]}))


def test_canonicalize_uses_given_map_when_valid():
    raw = pd.DataFrame({"x1": ["A"], "x2": ["2020-01-01"], "x3": ["4"]})
    cmap = {STATION: "x1", DATE: "x2", TARGET: "x3"}
    df, _ = canonicalize(raw, cmap)
    assert df[TARGET].iloc[0] == 4.0


def test_station_coverage_and_eligibility():
    d1 = pd.date_range("2000-01-01", periods=3000)               # complete, ~8.2 years
    d2 = pd.date_range("2000-01-01", periods=3000)[::10]          # 10% coverage
    df = pd.concat([pd.DataFrame({STATION: "full", DATE: d1, TARGET: np.arange(3000.0)}),
                    pd.DataFrame({STATION: "sparse", DATE: d2, TARGET: 1.0})])
    cov = station_coverage(df).set_index(STATION)
    assert cov.loc["full", "coverage"] == 1.0
    assert cov.loc["full", "max_missing_days_in_a_row"] == 0
    assert cov.loc["sparse", "max_missing_days_in_a_row"] == 9
    assert eligible_stations(cov.reset_index()) == ["full"]
