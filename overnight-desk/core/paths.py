from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
RAW = DATA / "raw"
CURATED = DATA / "curated"
REFERENCE = DATA / "reference"
LEDGER_DB = DATA / "ledger.db"

CONFIGS = ROOT / "configs"
ARTIFACTS = ROOT / "artifacts"
BRIEFINGS = ROOT / "briefings"


def ensure_dirs() -> None:
    for p in (RAW, CURATED, REFERENCE, ARTIFACTS, BRIEFINGS):
        p.mkdir(parents=True, exist_ok=True)
