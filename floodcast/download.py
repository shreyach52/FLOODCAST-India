"""Small, dependency-free file downloader (atomic, retrying, rejects HTML error pages)."""
from __future__ import annotations

import time
import urllib.request
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0 (floodcast-india hackathon project)"}


def download_file(url: str, dest: Path, retries: int = 3, timeout: int = 120,
                  force: bool = False) -> str:
    """Download url -> dest. Returns 'skipped' or 'downloaded'. Raises on failure."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return "skipped"
    part = dest.with_suffix(dest.suffix + ".part")
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r, open(part, "wb") as f:
                first = r.read(2048)
                head = first.lstrip()[:15].lower()
                if head.startswith(b"<!doctype") or head.startswith(b"<html"):
                    raise ValueError("Server returned an HTML page, not a data file "
                                     "(login wall or wrong link).")
                f.write(first)
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            part.replace(dest)
            return "downloaded"
        except Exception as e:  # noqa: BLE001
            last_err = e
            if part.exists():
                part.unlink()
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to download {url}: {last_err}")
