"""Threshold-free comparison of early-warning scores (pure numpy/pandas, no sklearn needed).

Why: onset_recall/onset_precision depend on ONE alarm rule, so a cautious model and a weak model
look alike. Here every model gives a continuous SCORE and we compare them over ALL alarm levels.

Onset event (same definition as the evaluation protocol): the river is BELOW its high-flow
threshold today (q0 < thr) and the true discharge h days later is AT/ABOVE it (y >= thr).
Score of a model = predicted discharge h days ahead / thr  (bigger = more alarming).
Persistence gets a real score too: q0 / thr  (its fixed-rule onset recall is 0 only by construction).

Metrics (all computed on onset-eligible rows only):
  roc_auc   chance a random onset row is scored above a random non-onset row (0.5 = coin flip)
  pr_auc    average precision; read it against the base rate `prevalence`
  tpr@F     share of onsets caught when only F of the non-onset rows may raise a false alarm
Uncertainty: station-block bootstrap (whole stations are resampled, because days of one station
are not independent), including a PAIRED difference between two models on the same resamples.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FPR_LEVELS = (0.01, 0.02, 0.05)


# ---------------------------------------------------------------- basic metrics
def roc_auc(y_true, score) -> float:
    y = np.asarray(y_true, bool)
    s = np.asarray(score, float)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _curve(y_true, score):
    """Cumulative (tp, fp) at the end of each group of tied scores, best score first."""
    y = np.asarray(y_true, bool)
    s = np.asarray(score, float)
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(~ys)
    last = np.r_[np.where(np.diff(ss) != 0)[0], len(ss) - 1]
    return tp[last], fp[last], int(y.sum()), int((~y).sum())


def average_precision(y_true, score) -> float:
    if len(y_true) == 0 or int(np.asarray(y_true, bool).sum()) == 0:
        return float("nan")
    tp, fp, n1, _ = _curve(y_true, score)
    precision = tp / (tp + fp)
    recall = tp / n1
    prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - prev) * precision))


def tpr_at_fpr(y_true, score, fpr_level: float) -> float:
    if len(y_true) == 0:
        return float("nan")
    tp, fp, n1, n0 = _curve(y_true, score)
    if n1 == 0 or n0 == 0:
        return float("nan")
    ok = np.where(fp / n0 <= fpr_level)[0]
    return float(tp[ok[-1]] / n1) if len(ok) else 0.0


def metric_value(name: str, y_true, score) -> float:
    if name == "roc_auc":
        return roc_auc(y_true, score)
    if name == "pr_auc":
        return average_precision(y_true, score)
    if name.startswith("tpr@"):
        return tpr_at_fpr(y_true, score, float(name[4:]))
    raise ValueError(f"unknown metric {name!r}")


METRIC_NAMES = ["roc_auc", "pr_auc"] + [f"tpr@{f}" for f in FPR_LEVELS]


# ---------------------------------------------------------------- data preparation
def prepare(df: pd.DataFrame, models: dict, q0="q0", y="y", thr="thr") -> pd.DataFrame:
    """Keep onset-eligible rows, add the label and one score column per model.

    models: {model_name: column holding that model's predicted discharge at t+h}.
    A 'persistence' score (q0/thr) is always added.
    """
    need = [q0, y, thr] + list(models.values())
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"columns not found: {miss}. Available: {list(df.columns)}")
    d = df[(df[q0] < df[thr]) & (df[thr] > 0)].copy()
    d["_label"] = d[y] >= d[thr]
    d["_s_persistence"] = d[q0] / d[thr]
    for name, col in models.items():
        d[f"_s_{name}"] = d[col] / d[thr]
    d = d.dropna(subset=["_label", "_s_persistence"] + [f"_s_{n}" for n in models])
    return d


def score_names(d: pd.DataFrame) -> list:
    return [c[3:] for c in d.columns if c.startswith("_s_")]


# ---------------------------------------------------------------- summary tables
def summarize(d: pd.DataFrame) -> pd.DataFrame:
    """One row per model on an already prepared frame."""
    rows = []
    n_pos = int(d["_label"].sum())
    for name in score_names(d):
        s = d[f"_s_{name}"].to_numpy()
        row = {"model": name, "n_rows": len(d), "n_onsets": n_pos,
               "prevalence": round(n_pos / max(len(d), 1), 4)}
        for m in METRIC_NAMES:
            row[m] = round(metric_value(m, d["_label"], s), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_by(d: pd.DataFrame, by=("horizon", "split")) -> pd.DataFrame:
    out = []
    for key, g in d.groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        t = summarize(g)
        for k, c in zip(by, key):
            t.insert(0, k, c)
        out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ---------------------------------------------------------------- station-block bootstrap
def paired_diff_ci(d: pd.DataFrame, model_a: str, model_b: str, metric: str = "roc_auc",
                   n_boot: int = 300, seed: int = 0, station: str = "station_id") -> dict:
    """metric(A) - metric(B) with a station-block bootstrap 95% interval.
    `prob_a_better` is the share of resamples where A beats B (not a p-value)."""
    y = d["_label"].to_numpy()
    sa, sb = d[f"_s_{model_a}"].to_numpy(), d[f"_s_{model_b}"].to_numpy()
    point = metric_value(metric, y, sa) - metric_value(metric, y, sb)
    codes, uniq = pd.factorize(d[station])
    idx = [np.flatnonzero(codes == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        rows = np.concatenate([idx[i] for i in pick])
        yy = y[rows]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        diffs.append(metric_value(metric, yy, sa[rows]) - metric_value(metric, yy, sb[rows]))
    diffs = np.array(diffs, float)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        return {"diff": point, "lo": np.nan, "hi": np.nan, "prob_a_better": np.nan, "n_boot": 0}
    return {"diff": round(point, 4), "lo": round(float(np.percentile(diffs, 2.5)), 4),
            "hi": round(float(np.percentile(diffs, 97.5)), 4),
            "prob_a_better": round(float((diffs > 0).mean()), 3), "n_boot": len(diffs)}


def compare_pairs(d: pd.DataFrame, pairs, metrics=("roc_auc", "tpr@0.02"), by=("horizon", "split"),
                  n_boot: int = 300, seed: int = 0) -> pd.DataFrame:
    rows = []
    for key, g in d.groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        for a, b in pairs:
            for m in metrics:
                r = paired_diff_ci(g, a, b, m, n_boot, seed)
                rows.append({**dict(zip(by, key)), "A": a, "B": b, "metric": m, **r})
    return pd.DataFrame(rows)
