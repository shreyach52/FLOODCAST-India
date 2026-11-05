"""Discharge QC rules (pandas/numpy only, unit-tested). See scripts/03b_qc_outliers.py.

DROP C  value > ABS_CAP (50,000 m3/s): physically implausible here, probably a unit (cusecs) or typing error.
DROP A  value > 20 x the station's own 99.9th percentile (typos such as 850,891 m3/s).
DROP B  isolated spike RUN: 1-3 consecutive OBSERVATIONS (missing days are skipped, span <= 5 days), each >= 40x the
        local reference and >= 10,000 m3/s, with the observation before and after the run both <= 25 % of the run's
        smallest value. A real flood rises and recedes
        over days; a value that appears and vanishes inside 1-3 days at that size is a recording/unit error.
        (Catches KOLLEGAL's consecutive pairs, which a one-day-spike rule misses.)
REVIEW  value > 30 x the +-3 day neighbour median and > 500 that is not dropped: kept, listed for a human.
local reference = max(21-day centred rolling median, station median of positive flows).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RATIO_B, MIN_B, MAX_RUN, ISO_FRAC, MAX_SPAN_DAYS = 40.0, 10_000.0, 3, 0.25, 5
ABS_CAP = 50_000.0   # judgement call (team may change): no peninsular river here is believed to reach this in m3/s


def _daily(s: pd.Series) -> pd.Series:
    return s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))


def flag_station(s: pd.Series) -> pd.DataFrame:
    """s: discharge indexed by date for ONE station. Returns rows [date, discharge, action, rule]."""
    q = _daily(s.sort_index())
    p999 = q.quantile(0.999)
    pos = q[q > 0]
    ref = q.rolling(21, center=True, min_periods=8).median().clip(lower=pos.median() if len(pos) else 1.0)
    out = {}

    # rule A
    a = q.notna() & (q > 20 * max(p999, 1e-9))
    for d in q.index[a]:
        out[d] = ("drop", "A_vs_own_p999")

    # rule C: absolute cap
    for d in q.index[q.notna() & (q > ABS_CAP)]:
        out.setdefault(d, ("drop", "C_above_abs_cap"))

    # rule B on the OBSERVED series (so a missing day inside a spike does not split it)
    obs = q.dropna()
    hot = ((q >= RATIO_B * ref) & (q >= MIN_B)).reindex(obs.index)
    grp = (hot != hot.shift()).cumsum()
    pos_i = pd.Series(np.arange(len(obs)), index=obs.index)
    for _, blk in pos_i[hot].groupby(grp[hot]):
        if len(blk) > MAX_RUN:
            continue
        lo, hi = blk.iloc[0], blk.iloc[-1]
        if (obs.index[hi] - obs.index[lo]).days > MAX_SPAN_DAYS:
            continue
        run_min = obs.iloc[lo:hi + 1].min()
        nb = [obs.iloc[i] for i in (lo - 1, hi + 1) if 0 <= i < len(obs)]
        if nb and all(v <= ISO_FRAC * run_min for v in nb):
            for d in obs.index[lo:hi + 1]:
                out.setdefault(d, ("drop", "B_isolated_spike_run"))

    # review rule
    nbmed = pd.concat([q.shift(k) for k in (-3, -2, -1, 1, 2, 3)], axis=1).median(axis=1)
    r = q.notna() & (q > 30 * nbmed.clip(lower=1e-9)) & (q > 500)
    for d in q.index[r]:
        out.setdefault(d, ("review", "R_spike_vs_neighbours"))

    if not out:
        return pd.DataFrame(columns=["date", "discharge", "action", "rule"])
    rows = [(d, q.loc[d], act, rule) for d, (act, rule) in sorted(out.items())]
    return pd.DataFrame(rows, columns=["date", "discharge", "action", "rule"])
