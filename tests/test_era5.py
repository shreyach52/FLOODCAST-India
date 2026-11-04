from datetime import date

import numpy as np

from floodcast.era5 import build_request, nearest_valid_cells, plan_chunks


def test_plan_chunks_current_year_only_complete_months():
    ch = plan_chunks(2024, today=date(2026, 11, 4))
    tags = [c[0] for c in ch]
    assert "2024" in tags and "2025" in tags
    assert "2026_09" in tags and "2026_10" not in tags and "2026" not in tags


def test_build_request_shape():
    r = build_request(2020, [1, 2])
    assert r["time"] == ["00:00"] and r["month"] == ["01", "02"] and len(r["day"]) == 31
    assert r["area"][0] > r["area"][2]            # north > south


def test_nearest_valid_cell_skips_sea():
    lat, lon = np.array([10.0, 10.1, 10.2]), np.array([75.0, 75.1, 75.2])
    valid = np.ones((3, 3), bool); valid[:, 0] = False          # western column is sea
    iy, ix, km = nearest_valid_cells([10.1], [75.0], lat, lon, valid)
    assert ix[0] == 1 and iy[0] == 1 and km[0] > 0