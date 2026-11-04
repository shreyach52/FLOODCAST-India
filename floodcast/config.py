"""Paths and the CWC download registry (resource ids copied from the NWDP dataset page)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
RAW_CWC = DATA_RAW / "cwc"
INTERIM = ROOT / "data" / "interim"
REPORTS = ROOT / "reports"
CONFIGS = ROOT / "configs"
COLUMN_MAP_PATH = CONFIGS / "column_map.json"

NWDP_BASE = "https://nwdp.nwic.gov.in"
CWC_DATASET_ID = "08fa3fd0-7861-471d-a295-27c1b239d1fa"

STATE_NAMES = {
    "AP": "Andhra Pradesh", "CG": "Chhattisgarh", "DD": "Daman & Diu", "GA": "Goa",
    "GJ": "Gujarat", "JH": "Jharkhand", "KA": "Karnataka", "KL": "Kerala",
    "MH": "Maharashtra", "ML": "Meghalaya", "MP": "Madhya Pradesh", "MZ": "Mizoram",
    "OD": "Odisha", "PY": "Puducherry", "RJ": "Rajasthan", "TN": "Tamil Nadu",
    "TR": "Tripura", "TS": "Telangana", "WB": "West Bengal",
}

# era -> filename pattern (note: the 1950-2000 files use a slightly different prefix)
ERA_FILENAME = {
    "1950_2000": "riverdischarge_manual_daily_cwc_{st}_1950_2000.csv",
    "2001_2025": "river_discharge_manual_daily_cwc_{st}_2001_2025.csv",
    "2026_2030": "river_discharge_manual_daily_cwc_{st}_2026_2030.csv",
}

# era -> state code (lowercase) -> resource id
CWC_RESOURCES = {
    "1950_2000": {
        "ap": "7a62a060-769a-4a96-afb4-329b4c78951a", "cg": "33129841-43dc-4b1f-9e03-47e236f3e96f",
        "gj": "3a86b991-7dd5-40db-bef1-ae5f6f35a211", "jh": "d8fa8ea5-237b-4c8c-9fe8-557e46aea75a",
        "ka": "e75d9aec-196b-4f68-8abe-255a5cd6376e", "kl": "7132d8ae-3174-408d-8ef3-37aa00752787",
        "mh": "1d7caf25-c30b-481e-8c10-bf8355ac3f98", "mp": "35648095-54a2-4a87-9fc3-1e437cf9179b",
        "od": "7a2eb488-8a47-4483-b593-a15843dafe9a", "py": "9dc9cc48-7dbd-4a7e-9292-d1b182eab123",
        "rj": "1f983f59-c157-4b89-af17-7d63bccd4a47", "tn": "cba07162-d1da-4987-b52e-0c1e2173e287",
        "ts": "e206b2fd-24b6-490f-90e5-e19eb87c259e", "wb": "3e38c530-39f1-4804-bba4-ccab7b70b36e",
    },
    "2001_2025": {
        "ap": "7f8eaf8a-a763-4456-a64d-350e55997ad9", "cg": "325ace25-9216-4701-95c5-b2027c5fb66c",
        "dd": "334a223a-24d8-436b-867e-213396e9644e", "ga": "27063a04-1f75-4447-a741-e189a48e1925",
        "gj": "b076861e-1513-410e-bafa-a1e34cd8f493", "jh": "29fddd3b-2ff0-4bcb-99aa-9234aa6778aa",
        "ka": "f95150ea-c8fc-4740-8815-d9c34c9d53a3", "kl": "c3cbe74d-2442-4496-aab2-21e8b5faa12e",
        "mh": "9c659865-ab21-4ffa-a3f9-edbae14f5c86", "ml": "0271ab76-01bc-4798-bf4e-45567b5e3f9b",
        "mp": "5708264d-5aea-4e39-8e64-e837f55d4c1b", "mz": "4126e89b-3a47-48c4-9783-65f00a4894ad",
        "od": "1286575d-3616-499a-a9db-45ed33f103e8", "py": "93e4a9c4-7871-49ef-bbdc-6391c2271403",
        "rj": "e62882fe-1b5e-4d17-8bec-62d537ea2a28", "tn": "fca9df0b-47b1-4f1a-8e59-1b43a8c0ae73",
        "tr": "c3bcd659-a544-4902-98b5-dd6248abf630", "ts": "1b9088b5-d196-4c5d-8780-a888e7e9e86b",
        "wb": "f414a75a-f71d-4522-9600-ed9c952fbacd",
    },
    "2026_2030": {
        "ap": "7a94f96b-7df5-4285-839d-c2dc8ba08241", "cg": "06a59e6c-8ebc-4dcc-a5de-894d078b5cbb",
        "dd": "223410c9-0e24-442f-9aad-bd8c102be68c", "ga": "f332f256-3dea-407c-8224-d3a43f1a5995",
        "gj": "0b648633-a893-4baf-9ab7-d42dbd80bb53", "jh": "b2ddfd83-ab81-4e76-91c3-4499506cc0e4",
        "ka": "8116803c-3edf-4c53-9840-19446ce857e7", "kl": "4c771ce1-6646-4dbc-9984-e40a3d5bd21f",
        "mh": "65e0ff3a-2b79-4e0b-b1ad-c2977cda9d09", "ml": "ad34e1ed-8961-46ad-984b-65044af7774b",
        "mp": "dbb429ad-6030-46d8-931a-cd3b1c1dedbd", "mz": "737171da-ed31-4afb-91c4-7d5166bc755c",
        "od": "1e501a44-ff95-42db-81e1-efa2be8be4dd", "py": "aca79e55-da2c-4e4c-bb0b-8249f6089e3d",
        "rj": "2d434fd7-4d0e-4b71-ab38-9c9716467175", "tn": "38543447-b022-4061-b246-d9c00d134639",
        "tr": "754fd532-1881-42d1-9b3b-5207e02c9141", "ts": "083774d3-ba42-44c5-977a-2965cb572354",
        "wb": "7dac0aa8-966a-4bd8-b347-2dc85c444931",
    },
}


def cwc_filename(state: str, era: str) -> str:
    return ERA_FILENAME[era].format(st=state.lower())


def cwc_url(state: str, era: str, base: str = NWDP_BASE) -> str:
    rid = CWC_RESOURCES[era][state.lower()]
    return f"{base}/dataset/{CWC_DATASET_ID}/resource/{rid}/download/{cwc_filename(state, era)}"
