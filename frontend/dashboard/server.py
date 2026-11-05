"""FLOODCAST INDIA dashboard: a tiny READ-ONLY API + static frontend. Python standard library + pandas only.

  python dashboard/server.py            ->  http://127.0.0.1:8000

It only reads files the backend already wrote. It never trains, predicts or edits anything.
"""
import os
import json
import mimetypes
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import logic  # noqa: E402
from floodcast import config  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"


class State:
    def __init__(self):
        self.cfg = logic.load_cfg(config.CONFIGS / "frontend.json")
        path = Path(config.INTERIM) / "discharge_daily.csv.gz"
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run the data pipeline (scripts 01-03) first.")
        self.df = pd.read_csv(path, parse_dates=["date"])
        self.thr = logic.thresholds(self.df, self.cfg)
        self.table = logic.station_table(self.df, self.thr, self.cfg)
        self.updated = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        self._cache = {}

    def optional_csv(self, key, **kw):
        """Read an optional backend output; re-read only if the file changed. Returns None if absent."""
        p = ROOT / self.cfg[key]
        if not p.exists():
            return None
        m = p.stat().st_mtime
        if self._cache.get(key, (None,))[0] != m:
            self._cache[key] = (m, pd.read_csv(p, **kw))
        return self._cache[key][1]


S = None


def api_summary():
    t, cfg = S.table, S.cfg
    fresh = t[t.fresh]
    stations = []
    for sid, r in t.iterrows():
        stations.append({"id": sid, "river": logic.txt(r.get("river")), "state": logic.txt(r.get("state")),
                         "lat": logic.num(r.get("lat"), 4), "lon": logic.num(r.get("lon"), 4),
                         "last_date": r.last_date.strftime("%Y-%m-%d"), "last_q": logic.num(r.last_q),
                         "elevated": logic.num(r.elevated), "high": logic.num(r.high),
                         "fresh": bool(r.fresh), "risk": logic.txt(r.risk)})
    counts = {k: int((t.risk == k).sum()) for k in logic.RISKS}
    data_max = S.df["date"].max()
    return {
        "n_stations": int(len(t)), "n_reporting": int(len(fresh)),
        "data_last_date": data_max.strftime("%Y-%m-%d"),
        "n_on_last_date": int((t.last_date == data_max).sum()),
        "file_updated": S.updated, "counts": counts,
        "n_without_threshold": int(t.high.isna().sum()),
        "above_high": counts["High"],
        "fresh_days": cfg["fresh_days"], "high_quantile": cfg["high_quantile"],
        "elevated_quantile": cfg["elevated_quantile"], "train_end": cfg["train_end"],
        "forecasts_available": (ROOT / cfg["forecast_file"]).exists(),
        "weather_available": (ROOT / cfg["weather_file"]).exists(),
        "stations": stations,
    }


def api_station(sid, days):
    g = S.df[S.df.station_id == sid]
    if g.empty:
        return None
    dates, q = logic.daily_window(g, days)
    r = S.table.loc[sid]
    return {"id": sid, "river": logic.txt(r.get("river")), "state": logic.txt(r.get("state")),
            "lat": logic.num(r.get("lat"), 4), "lon": logic.num(r.get("lon"), 4),
            "first_date": g.date.min().strftime("%Y-%m-%d"), "n_obs": int(len(g)),
            "last_date": r.last_date.strftime("%Y-%m-%d"), "last_q": logic.num(r.last_q),
            "elevated": logic.num(r.elevated), "high": logic.num(r.high), "risk": logic.txt(r.risk),
            "dates": dates, "discharge": q}


def api_forecasts(sid):
    fc = S.optional_csv("forecast_file")
    if fc is None:
        return {"available": False, "reason": f"File not found: {S.cfg['forecast_file']}"}
    fc = fc[fc.station_id == sid]
    if fc.empty:
        return {"available": False, "reason": "No forecasts stored for this station."}
    obs = S.df[S.df.station_id == sid].set_index("date")["discharge"]
    out = {}
    for h in (1, 3, 7):
        a = logic.align_forecasts(fc, obs, h)
        if a:
            out[str(h)] = a
    return {"available": bool(out), "horizons": out,
            "reason": None if out else "No 1/3/7-day forecasts stored for this station."}


def api_weather(sid):
    w = S.optional_csv("weather_file", parse_dates=["date"])
    if w is None:
        return {"available": False, "reason": f"File not found: {S.cfg['weather_file']} (run scripts 06a and 06b)."}
    w = w[w.station_id == sid]
    if w.empty:
        return {"available": False, "reason": "No weather rows for this station."}
    return {"available": True, **logic.weather_summary(w)}


def api_metrics():
    frames = []
    for f in S.cfg["metrics_files"]:
        p = ROOT / f
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        return {"available": False, "reason": "No metrics file found (run scripts/05_baseline.py)."}
    m = pd.concat(frames, ignore_index=True).drop_duplicates(["horizon", "split", "model"], keep="last")
    rows = [{k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in r.items()}
            for r in m.to_dict("records")]
    return {"available": True, "rows": rows, "models": sorted(m.model.unique().tolist())}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, allow_nan=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        parts = [unquote(p) for p in u.path.strip("/").split("/")]
        try:
            if parts[0] == "api":
                return self._api(parts[1:], parse_qs(u.query))
            return self._static(u.path)
        except Exception as e:  # never crash the server; show the reason in the UI
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def _api(self, p, q):
        if p == ["summary"]:
            return self._json(api_summary())
        if p == ["metrics"]:
            return self._json(api_metrics())
        if len(p) == 2 and p[0] in ("station", "forecasts", "weather"):
            if p[1] not in S.table.index:
                return self._json({"error": f"Unknown station: {p[1]}"}, 404)
            if p[0] == "station":
                days = max(30, min(int(q.get("days", ["365"])[0]), 3650))
                return self._json(api_station(p[1], days))
            return self._json(api_forecasts(p[1]) if p[0] == "forecasts" else api_weather(p[1]))
        return self._json({"error": "Not found"}, 404)

    def _static(self, path):
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        f = (STATIC / rel).resolve()
        if STATIC.resolve() not in f.parents or not f.is_file():
            self.send_response(404)
            self.end_headers()
            return
        body = f.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", (mimetypes.guess_type(f.name)[0] or "application/octet-stream") + "; charset=utf-8"
                         if f.suffix in (".html", ".css", ".js") else (mimetypes.guess_type(f.name)[0] or "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    global S
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    S = State()
    print(f"FLOODCAST INDIA dashboard -> http://127.0.0.1:{a.port}   (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

