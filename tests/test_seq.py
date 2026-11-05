import numpy as np
import pandas as pd

from floodcast.evalcommon import H
from floodcast.seq import build_arrays, evaluate, gather, residual_targets, select, standardise_static

TR, VA, TE = "2010-12-31", "2012-12-31", "2013-01-01"


def _data(nst=4, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2008-01-01", "2014-12-31")
    d, w = [], []
    for k in range(nst):
        q = np.exp(rng.normal(3 + k, 0.5, len(idx)))
        d.append(pd.DataFrame({"station_id": f"S{k}", "date": idx, "discharge": q, "lat": 10 + k, "lon": 75 + k}))
        w.append(pd.DataFrame({"station_id": f"S{k}", "date": idx, "rain_mm": rng.gamma(0.3, 8, len(idx)),
                               "soil1": 0.2 + 0.1 * rng.random(len(idx))}))
    return pd.concat(d), pd.concat(w)


def test_window_uses_only_past_and_targets_align():
    df, w = _data()
    d = build_arrays(df, w, ["S0", "S1", "S2", "S3"], TR, L=30)
    k = np.nonzero(d.valid)[0][100]
    win = gather(d.A, np.array([k]), 30)
    A2 = d.A.copy(); A2[k + 1:k + 50] = 99.0                      # change the FUTURE of row k
    assert np.allclose(win, gather(A2, np.array([k]), 30))
    for i, h in enumerate(H):                                     # y_h is discharge h days later
        j = k + h
        assert d.st_idx[j] == d.st_idx[k] and np.isclose(d.y[k, i], d.q0[j])


def test_windows_never_cross_station_boundary():
    df, w = _data()
    d = build_arrays(df, w, ["S0", "S1"], TR, L=30)
    first = {s: np.nonzero(d.st_idx == s)[0][0] for s in (0, 1)}
    assert not d.valid[first[1]:first[1] + 29].any() and not d.valid[first[0]:first[0] + 29].any()


def test_standardisation_uses_train_rows_only():
    df, w = _data()
    a = build_arrays(df, w, ["S0", "S1"], TR, L=30)
    df2 = df.copy(); df2.loc[df2.date > TR, "discharge"] *= 50    # change post-train data
    b = build_arrays(df2, w, ["S0", "S1"], TR, L=30)
    tr_rows = a.date <= np.datetime64(pd.Timestamp(TR))
    assert np.allclose(a.A[tr_rows][:, 0], b.A[tr_rows][:, 0])   # q_anom on train rows unchanged


def test_masks_respect_split_boundaries_and_holdout():
    df, w = _data()
    d = build_arrays(df, w, ["S0", "S1", "S2", "S3"], TR, L=30)
    held = {"S3"}
    m = select(d, held, TR, VA, TE)
    tdate = d.date[:, None] + np.array(H, dtype="timedelta64[D]")[None, :]
    assert (tdate[m["train"]] <= np.datetime64(pd.Timestamp(TR))).all()      # no target beyond train end
    assert (d.date[m["A_seen_stations"].any(1)] >= np.datetime64(pd.Timestamp(TE))).all()
    sid = np.array(d.names)[d.st_idx]
    assert not np.isin(sid[m["train"].any(1)], list(held)).any()             # held-out never trained on
    assert set(sid[m["B_unseen_stations"].any(1)]) == held


def test_evaluate_zero_residual_equals_persistence():
    df, w = _data()
    d = build_arrays(df, w, ["S0", "S1", "S2", "S3"], TR, L=30)
    m = select(d, {"S3"}, TR, VA, TE)
    res, preds = evaluate(d, m, np.zeros((len(d.q0), 3)))
    a = res[res.model == "persistence"].reset_index(drop=True); b = res[res.model == "lstm"].reset_index(drop=True)
    assert np.allclose(a.rmse, b.rmse) and len(a) == 6 and len(preds) > 0
    assert residual_targets(d).shape == (len(d.q0), 3)
    assert standardise_static(d.static, d.names, {"S3"}).shape == (4, 3)


def test_report_tables_with_and_without_baseline(tmp_path):
    from floodcast.evalcommon import combine_with_baseline, delta_table
    df, w = _data()
    d = build_arrays(df, w, ["S0", "S1", "S2", "S3"], TR, L=30)
    res, _ = evaluate(d, select(d, {"S3"}, TR, VA, TE), np.zeros((len(d.q0), 3)))
    allr = combine_with_baseline(res, tmp_path / "missing.csv")          # no baseline file yet
    assert set(allr.model) == {"persistence", "lstm"} and delta_table(allr)[0] is None
    b = res[res.model == "lstm"].assign(model="lightgbm")                # fake baseline without weather rows
    b.to_csv(tmp_path / "b.csv", index=False)
    allr = combine_with_baseline(res, tmp_path / "b.csv")
    delta, ref = delta_table(allr)
    assert ref == "lightgbm" and (delta.abs().fillna(0) < 1e-9).all().all()


def test_station_groups_late_start_is_unseen_and_uses_val_reference():
    from floodcast.evalcommon import station_groups
    df, _ = _data(nst=5)
    df = df[~((df.station_id == "S4") & (df.date <= "2011-06-30"))]          # S4 starts after train end (2010-12-31)
    df = df[~((df.station_id == "S3") & (df.date > "2008-02-15"))]           # S3 has <30 obs ever past 2008-02
    g = station_groups(df, {"S0", "S1", "S2", "S3", "S4"}, TR, VA, frac=0.5, seed=1)
    assert "S4" in g["late"] and "S4" in g["unseen"] and "S4" not in g["hist"]
    assert pd.Timestamp(g["ref_end"]["S4"]) == pd.Timestamp(VA) and pd.Timestamp(g["ref_end"]["S0"]) == pd.Timestamp(TR)
    assert g["held"] <= g["hist"] and not (g["held"] & g["late"])
