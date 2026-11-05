import subprocess
import sys
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_onset_eval import _synthetic

ROOT = Path(__file__).resolve().parents[1]


def test_cli_end_to_end(tmp_path):
    df = _synthetic()
    df = df.rename(columns={"p_good": "pred_lstm", "p_weak": "pred_lgb", "p_cautious": "pred_x"})
    f1 = tmp_path / "a.csv.gz"
    df.drop(columns=["pred_lgb"]).to_csv(f1, index=False)
    f2 = tmp_path / "b.csv.gz"
    df[["station_id", "date", "horizon", "pred_lgb"]].to_csv(f2, index=False)
    out_md, out_csv = tmp_path / "r.md", tmp_path / "s.csv"
    cmd = [sys.executable, str(ROOT / "scripts" / "08_onset_curves.py"), "--pred", str(f1),
           "--model", "lstm=pred_lstm", "--pred2", str(f2), "--model2", "lgb=pred_lgb",
           "--n-boot", "20", "--out-md", str(out_md), "--out-csv", str(out_csv)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    text = out_md.read_text()
    assert "roc_auc" in text and "persistence" in text and "lgb" in text
    assert out_csv.exists()


def test_cli_wrong_column_prints_real_header(tmp_path):
    f = tmp_path / "a.csv"
    _synthetic().to_csv(f, index=False)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "08_onset_curves.py"), "--pred", str(f),
                        "--model", "m=does_not_exist"], capture_output=True, text=True)
    assert r.returncode != 0 and "Columns in the file" in (r.stderr + r.stdout)


def test_cli_same_column_name_in_both_files(tmp_path):
    df = _synthetic().rename(columns={"p_good": "pred", "p_weak": "other"})
    f1 = tmp_path / "a.csv"
    df.drop(columns=["p_cautious"]).to_csv(f1, index=False)
    f2 = tmp_path / "b.csv"
    df2 = df[["station_id", "date", "horizon"]].copy()
    df2["pred"] = df["other"]                                    # file 2 'pred' is the WEAK model
    df2.to_csv(f2, index=False)
    out_csv = tmp_path / "s.csv"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "08_onset_curves.py"), "--pred", str(f1),
                        "--model", "first=pred", "--pred2", str(f2), "--model2", "second=pred",
                        "--n-boot", "10", "--out-md", str(tmp_path / "r.md"), "--out-csv", str(out_csv)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    import pandas as pd
    t = pd.read_csv(out_csv).groupby("model").roc_auc.mean()
    assert t["first"] > t["second"] + 0.05      # proves each name kept its own column
