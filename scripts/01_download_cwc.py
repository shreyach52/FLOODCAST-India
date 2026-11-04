"""Download CWC daily discharge CSVs from the National Water Data Portal.

Examples:
  python scripts/01_download_cwc.py --states KA KL            # Karnataka + Kerala, 2001-2025 and 2026-2030
  python scripts/01_download_cwc.py --states KA --eras 1950_2000 2001_2025
  python scripts/01_download_cwc.py --all
"""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

from floodcast import config
from floodcast.download import download_file


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--states", nargs="+", help="state codes, e.g. KA KL MH")
    g.add_argument("--all", action="store_true", help="every state available per era")
    ap.add_argument("--eras", nargs="+", default=["2001_2025", "2026_2030"],
                    choices=list(config.ERA_FILENAME))
    ap.add_argument("--out", default=str(config.RAW_CWC))
    ap.add_argument("--base-url", default=config.NWDP_BASE, help="for testing only")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    jobs = []
    for era in a.eras:
        avail = config.CWC_RESOURCES[era]
        wanted = list(avail) if a.all else [s.lower() for s in a.states]
        for st in wanted:
            if st not in avail:
                print(f"[skip] {st.upper()} has no file for era {era}")
                continue
            jobs.append((st, era))

    ok = fail = 0
    for st, era in jobs:
        url = config.cwc_url(st, era, a.base_url)
        dest = Path(a.out) / config.cwc_filename(st, era)
        try:
            res = download_file(url, dest, force=a.force)
            size = dest.stat().st_size / 1e6
            print(f"[{res}] {dest.name} ({size:.1f} MB)")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"[FAILED] {st.upper()} {era}: {e}")
            fail += 1
    print(f"\nDone. {ok} ok, {fail} failed. Files are in {a.out}")
    if fail:
        print("If a download fails, open the link in your browser, save the CSV into that "
              "folder manually, and carry on.")


if __name__ == "__main__":
    main()
