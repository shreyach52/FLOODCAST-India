import functools
import http.server
import threading

import pytest

from floodcast import config
from floodcast.download import download_file


@pytest.fixture()
def server(tmp_path):
    root = tmp_path / "www"
    root.mkdir()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield root, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_download_and_skip(server, tmp_path):
    root, base = server
    (root / "a.csv").write_text("x,y\n1,2\n")
    dest = tmp_path / "out" / "a.csv"
    assert download_file(f"{base}/a.csv", dest) == "downloaded"
    assert dest.read_text() == "x,y\n1,2\n"
    assert download_file(f"{base}/a.csv", dest) == "skipped"
    assert not list(dest.parent.glob("*.part"))


def test_html_page_is_rejected(server, tmp_path):
    root, base = server
    (root / "login.csv").write_text("<!DOCTYPE html><html>please log in</html>")
    with pytest.raises(RuntimeError):
        download_file(f"{base}/login.csv", tmp_path / "x.csv", retries=1)
    assert not (tmp_path / "x.csv").exists()


def test_missing_file_fails_cleanly(server, tmp_path):
    _, base = server
    with pytest.raises(RuntimeError):
        download_file(f"{base}/nope.csv", tmp_path / "n.csv", retries=1)


def test_registry_is_complete_and_urls_look_right():
    for era, table in config.CWC_RESOURCES.items():
        assert set(table) <= {k.lower() for k in config.STATE_NAMES}
        for st, rid in table.items():
            assert len(rid) == 36
            url = config.cwc_url(st, era)
            assert url.startswith(config.NWDP_BASE) and url.endswith(f"_{era}.csv")
    assert len(config.CWC_RESOURCES["2001_2025"]) == 19
    assert len(config.CWC_RESOURCES["1950_2000"]) == 14
    ids = [r for t in config.CWC_RESOURCES.values() for r in t.values()]
    assert len(ids) == len(set(ids)), "duplicate resource id"
