import numpy as np
import pandas as pd
import pytest

from floodcast.onset_eval import (average_precision, compare_pairs, paired_diff_ci, prepare,
                                  roc_auc, summarize, summarize_by, tpr_at_fpr)


def test_auc_ap_match_sklearn_including_ties():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    y = rng.random(2000) < 0.1
    s = np.round(rng.normal(size=2000) + 1.2 * y, 1)          # rounding creates many ties
    assert roc_auc(y, s) == pytest.approx(sk.roc_auc_score(y, s))
    assert average_precision(y, s) == pytest.approx(sk.average_precision_score(y, s))


def test_tpr_at_fpr_hand_example():
    # 2 onsets (scores .9,.5) and 8 non-onsets (scores .8,.4,.3,.3,.2,.2,.1,.0)
    y = [1, 1] + [0] * 8
    s = [.9, .5, .8, .4, .3, .3, .2, .2, .1, .0]
    assert tpr_at_fpr(y, s, 0.0) == 0.5            # only the .9 is above every false alarm
    assert tpr_at_fpr(y, s, 0.125) == 1.0          # one false alarm (.8) allowed -> cutoff .5 catches both


def _synthetic(n_st=12, n_days=300, seed=0):
    """Truth: onset happens when latent rises. 'good' sees the latent, 'weak' sees it noisily,
    'cautious' = good scaled down (same ranking!)."""
    rng = np.random.default_rng(seed)
    rows = []
    for st in range(n_st):
        thr = rng.uniform(50, 200)
        q0 = thr * rng.uniform(0.1, 0.95, n_days)
        latent = rng.normal(size=n_days)
        y = q0 * np.exp(0.6 * latent + 0.25)
        for h in (1, 3):
            rows.append(pd.DataFrame({
                "station_id": f"S{st}", "date": pd.date_range("2022-01-01", periods=n_days),
                "horizon": h, "split": "A", "q0": q0, "thr": thr, "y": y,
                "p_good": q0 * np.exp(0.6 * latent + 0.25),
                "p_weak": q0 * np.exp(0.6 * (latent + rng.normal(0, 2.5, n_days)) + 0.25),
                "p_cautious": 0.3 * q0 * np.exp(0.6 * latent + 0.25)}))
    # some rows already above threshold -> must be dropped
    df = pd.concat(rows, ignore_index=True)
    df.loc[df.index[:50], "q0"] = df.loc[df.index[:50], "thr"] * 1.5
    return df


MODELS = {"good": "p_good", "weak": "p_weak", "cautious": "p_cautious"}


def test_prepare_drops_rows_already_above_threshold_and_adds_persistence():
    df = _synthetic()
    d = prepare(df, MODELS)
    assert (d["q0"] < d["thr"]).all()
    assert len(d) == len(df) - 50
    assert "_s_persistence" in d.columns and (d["_s_persistence"] < 1).all()
    with pytest.raises(KeyError):
        prepare(df, {"x": "nope"})


def test_ranking_metrics_ignore_a_constant_rescaling_but_a_fixed_rule_would_not():
    d = prepare(_synthetic(), MODELS)
    t = summarize(d).set_index("model")
    assert t.loc["good", "roc_auc"] == pytest.approx(t.loc["cautious", "roc_auc"])
    assert t.loc["good", "roc_auc"] > t.loc["weak", "roc_auc"] > 0.5
    assert t.loc["good", "tpr@0.02"] > t.loc["weak", "tpr@0.02"]
    # fixed alarm rule 'score >= 1': the cautious model looks far worse although it ranks identically
    lab = d["_label"].to_numpy()
    rec = lambda m: ((d[f"_s_{m}"] >= 1) & lab).sum() / lab.sum()
    assert rec("cautious") < 0.5 * rec("good")


def test_persistence_has_a_real_score():
    t = summarize(prepare(_synthetic(), MODELS)).set_index("model")
    assert t.loc["persistence", "roc_auc"] > 0.5      # higher flow today -> closer to crossing


def test_summarize_by_groups():
    d = prepare(_synthetic(), MODELS)
    t = summarize_by(d, by=("horizon", "split"))
    assert set(t["horizon"]) == {1, 3} and len(t) == 2 * 4        # 3 models + persistence


def test_paired_bootstrap_sign_ci_and_reproducibility():
    d = prepare(_synthetic(), MODELS)
    r1 = paired_diff_ci(d, "good", "weak", "roc_auc", n_boot=100, seed=3)
    r2 = paired_diff_ci(d, "good", "weak", "roc_auc", n_boot=100, seed=3)
    assert r1 == r2 and r1["diff"] > 0 and r1["lo"] > 0 and r1["prob_a_better"] > 0.95
    same = paired_diff_ci(d, "good", "cautious", "roc_auc", n_boot=50, seed=1)
    assert same["diff"] == pytest.approx(0, abs=1e-9) and same["lo"] <= 0 <= same["hi"]


def test_compare_pairs_table():
    d = prepare(_synthetic(), MODELS)
    t = compare_pairs(d, [("good", "weak")], metrics=("roc_auc",), n_boot=30)
    assert len(t) == 2 and set(t["metric"]) == {"roc_auc"}
