"""STEP 4: multi-horizon LSTM (1, 3, 7 days ahead from ONE shared network), compared with the LightGBM baseline.

  python scripts/07_lstm.py --quick        # 2-minute pipeline check (tiny training). Do this first.
  python scripts/07_lstm.py                # real run (weather used if data/interim/weather_daily.csv.gz exists)
  python scripts/07_lstm.py --no-weather   # discharge-only LSTM (ablation)
  python scripts/07_lstm.py --hi-weight 3  # emphasise high-flow targets in the loss (more recall, more false alarms)
  python scripts/07_lstm.py --seeds 3      # average 3 trained networks (use for the final numbers)

Same stations, held-out set, split dates, residual target and metrics as 05_baseline.py (shared code in
floodcast/evalcommon.py and floodcast/seq.py). Input: the previous --window days of [discharge anomaly, observed-mask,
rain, soil moisture, day-of-year] ending on day t, plus static [log station median, lat, lon]. No future data is used.
Outputs: reports/lstm_report.md, data/interim/lstm_metrics.csv, data/interim/lstm_test_predictions.csv.gz
"""
import _bootstrap  # noqa: F401
import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd

from floodcast import config
from floodcast.evalcommon import H, combine_with_baseline, delta_table, ready_stations, station_groups
from floodcast.seq import build_arrays, evaluate, gather, residual_targets, select, standardise_static

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=str(config.INTERIM))
    ap.add_argument("--train-end", default="2018-12-31")
    ap.add_argument("--val-end", default="2021-12-31")
    ap.add_argument("--test-start", default="2022-01-01")
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42, help="holdout seed: keep 42 to match the baseline")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--weather", default=None)
    ap.add_argument("--no-weather", action="store_true")
    ap.add_argument("--window", type=int, default=60, help="history length in days")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-train", type=int, default=250_000, help="random training rows per epoch")
    ap.add_argument("--max-val", type=int, default=100_000)
    ap.add_argument("--hi-weight", type=float, default=1.0, help="loss weight for targets above the station's train p95")
    ap.add_argument("--seeds", type=int, default=1, help="number of networks to average")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.epochs, a.max_train, a.max_val = 2, 30_000, 20_000
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise SystemExit("PyTorch is not installed. Install the CPU build from https://pytorch.org (pip install torch). "
                         "If pip cannot find a wheel for your Python version, create a venv with Python 3.12.")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ind = Path(a.in_dir)
    df = pd.read_csv(ind / "discharge_daily.csv.gz", parse_dates=["date"])
    ready = ready_stations(ind, a.exclude)
    grp = station_groups(df, ready, a.train_end, a.val_end, a.holdout_frac, a.seed)
    held, ready = grp["unseen"], grp["hist"] | grp["late"]
    print(f"stations: {len(grp['hist']) - len(grp['held'])} trained-on, {len(grp['held'])} random held-out, "
          f"{len(grp['late'])} late-start (no training-period data), {len(grp['dropped'])} dropped (too little data)")
    wpath = Path(a.weather) if a.weather else ind / "weather_daily.csv.gz"
    use_wx = (not a.no_weather) and wpath.exists()
    w = pd.read_csv(wpath, parse_dates=["date"]) if use_wx else None
    if not a.no_weather and not use_wx:
        print(f"NOTE: {wpath} not found -> discharge-only LSTM.")
    d = build_arrays(df, w, ready, a.train_end, L=a.window, use_weather=use_wx, ref_end=grp["ref_end"])
    st_std = standardise_static(d.static, d.names, held)
    R = residual_targets(d)
    masks = select(d, held, a.train_end, a.val_end, a.test_start)
    F = d.A.shape[1]
    print(f"device={device} weather={use_wx} stations={len(d.names)} held-out={len(held)} features={d.feat_names} "
          f"train rows={int(masks['train'].any(1).sum()):,} val rows={int(masks['val'].any(1).sum()):,}")

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(F, a.hidden, a.layers, batch_first=True, dropout=a.dropout if a.layers > 1 else 0.0)
            self.head = nn.Sequential(nn.Linear(a.hidden + 3, 64), nn.ReLU(), nn.Dropout(a.dropout), nn.Linear(64, len(H)))

        def forward(self, x, st):
            out, _ = self.lstm(x)
            return self.head(torch.cat([out[:, -1], st], dim=1))

    hi = (d.y >= d.thr[:, None])

    def tens(rows):
        x = torch.from_numpy(gather(d.A, rows, a.window)).to(device)
        st = torch.from_numpy(st_std[d.st_idx[rows]]).to(device)
        return x, st

    def loss_fn(pred, rows, mask, weighted):
        r = torch.from_numpy(np.nan_to_num(R[rows])).to(device)
        m = torch.from_numpy(mask[rows].astype(np.float32)).to(device)
        if weighted and a.hi_weight != 1.0:
            m = m * (1 + (a.hi_weight - 1) * torch.from_numpy(hi[rows].astype(np.float32)).to(device))
        l = torch.nn.functional.smooth_l1_loss(pred, r, beta=0.5, reduction="none")
        return (l * m).sum() / m.sum().clamp(min=1.0)

    def predict(net, rows):
        net.eval()
        out = np.zeros((len(rows), len(H)), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(rows), 4096):
                x, st = tens(rows[i:i + 4096])
                out[i:i + 4096] = net(x, st).cpu().numpy()
        return out

    tr_rows = np.nonzero(masks["train"].any(1))[0]
    va_rows = np.nonzero(masks["val"].any(1))[0]
    test_rows = np.nonzero(masks["A_seen_stations"].any(1) | masks["B_unseen_stations"].any(1))[0]
    preds = []
    for s in range(a.seeds):
        seed = 1000 + s
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = Net().to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
        va_pick = rng.choice(va_rows, size=min(a.max_val, len(va_rows)), replace=False)
        best, best_state, bad = np.inf, None, 0
        for ep in range(a.epochs):
            t0 = time.time()
            net.train()
            pick = rng.choice(tr_rows, size=min(a.max_train, len(tr_rows)), replace=False)
            tot = 0.0
            for i in range(0, len(pick), a.batch):
                rows = pick[i:i + a.batch]
                x, st = tens(rows)
                loss = loss_fn(net(x, st), rows, masks["train"], True)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                tot += float(loss) * len(rows)
            net.eval()
            with torch.no_grad():
                vl = float(np.mean([float(loss_fn(net(*tens(va_pick[i:i + 4096])), va_pick[i:i + 4096], masks["val"], False))
                                    for i in range(0, len(va_pick), 4096)]))
            print(f"seed {s} epoch {ep + 1}/{a.epochs} train {tot / len(pick):.4f} val {vl:.4f} ({time.time() - t0:.0f}s)", flush=True)
            if vl < best - 1e-4:
                best, best_state, bad = vl, copy.deepcopy(net.state_dict()), 0
            else:
                bad += 1
                if bad >= a.patience:
                    break
        net.load_state_dict(best_state if best_state is not None else net.state_dict())
        best_state = best_state if best_state is not None else copy.deepcopy(net.state_dict())
        torch.save(best_state, ind / f"lstm_seed{s}.pt")
        preds.append(predict(net, test_rows))
    resid = np.zeros((len(d.q0), len(H)), dtype=np.float32)
    resid[test_rows] = np.mean(preds, axis=0)

    res, pred_long = evaluate(d, masks, resid, "lstm")
    res.to_csv(ind / "lstm_metrics.csv", index=False)
    pred_long.to_csv(ind / "lstm_test_predictions.csv.gz", index=False)

    allr = combine_with_baseline(res, ind / "baseline_metrics.csv")
    show = allr.round(3)
    md = [f"# LSTM report ({'discharge + weather' if use_wx else 'discharge-only'})\n",
          f"Window {a.window} d, hidden {a.hidden} x {a.layers} layers, hi-weight {a.hi_weight}, seeds {a.seeds}, "
          f"epochs<= {a.epochs}{' (QUICK run: not a real result)' if a.quick else ''}. "
          f"Train <= {a.train_end}, val <= {a.val_end}, test >= {a.test_start}. Unseen stations: {len(grp['held'])} random held-out {sorted(map(str, grp['held']))} + {len(grp['late'])} late-start {sorted(map(str, grp['late']))}\n",
          "lightgbm rows come from data/interim/baseline_metrics.csv (re-run 05_baseline.py with the SAME flags first, "
          "otherwise they are not comparable).\n", "```", show.to_string(index=False), "```"]
    delta, ref = delta_table(allr)
    if delta is not None:
        md += ["", f"## LSTM minus {ref} (positive = LSTM better)", "```", delta.to_string(), "```"]
    (config.REPORTS / "lstm_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
