import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
ROOT = Path(__file__).resolve().parents[1]


def make_world(dirpath, seed=0, n_st=16, n_late=2):
    """Rain really drives the river: q_t = 0.85 q_{t-1} + 1.2 rain_t + small noise (linear reservoir)."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2008-01-01", "2023-12-31")
    doy = days.dayofyear.to_numpy()
    wet_p = 0.08 + 0.55 * np.exp(-((doy - 200) ** 2) / (2 * 40.0 ** 2))
    dis, wx, win = [], [], []
    for i in range(n_st):
        wet = rng.random(len(days)) < wet_p
        rain = np.where(wet, rng.gamma(0.8, 14.0, len(days)), 0.0)
        q = np.zeros(len(days))
        q[0] = 5
        for t in range(1, len(days)):
            q[t] = 0.85 * q[t - 1] + 1.2 * rain[t] * (1 + 0.1 * i) + rng.normal(0, 0.3) + 0.3
        q = np.clip(q, 0, None)
        start = 0
        late = i >= n_st - n_late
        if late:
            start = int(np.flatnonzero(days >= "2019-06-01")[0])        # record begins mid-2019
        sid = f"S{i:02d}"
        dis.append(pd.DataFrame({"station_id": sid, "date": days[start:], "discharge": q[start:]}))
        wx.append(pd.DataFrame({"station_id": sid, "date": days, "rain_mm": rain,
                                "soil1": 0.2 + 0.001 * pd.Series(rain).rolling(10, min_periods=1).sum().to_numpy()}))
        win.append({"station_id": sid, "sequence_ready": True})
    dirpath.mkdir(parents=True, exist_ok=True)
    pd.concat(dis).to_csv(dirpath / "discharge_daily.csv.gz", index=False)
    pd.concat(wx).to_csv(dirpath / "weather_daily.csv.gz", index=False)
    pd.DataFrame(win).to_csv(dirpath / "station_windows.csv", index=False)


def test_more_forecast_skill_means_better_early_warning(tmp_path):
    make_world(tmp_path / "in")
    out_csv, out_md = tmp_path / "o.csv", tmp_path / "o.md"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "09_future_rain_value.py"),
                        "--in-dir", str(tmp_path / "in"), "--quick", "--rhos", "0.5", "0.9", "1.0",
                        "--out-csv", str(out_csv), "--out-md", str(out_md)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:] + r.stdout[-1000:]
    t = pd.read_csv(out_csv)
    assert set(t.model) >= {"persistence", "wx", "fc_rho0.5", "fc_rho0.9", "oracle"}
    assert set(t.split) == {"A", "B"}, "both seen (A) and unseen (B) stations must be evaluated"
    for h in (1, 3, 7):
        a = t[(t.horizon == h) & (t.split == "A")].set_index("model").roc_auc
        assert a["oracle"] > a["wx"] + 0.02, (h, a.to_dict())
        assert a["fc_rho0.9"] >= a["fc_rho0.5"] - 0.005, (h, a.to_dict())
        assert a["fc_rho0.5"] >= a["wx"] - 0.01, (h, a.to_dict())
    text = out_md.read_text()
    assert "NOT a deployable model" in text and "Gain over `wx`" in text
