from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter

router = APIRouter()

THESIS_PATH = Path(os.getenv("THESIS_PATH", "./thesis.txt"))


@router.get("/")
def get_thesis() -> dict[str, str]:
    if THESIS_PATH.exists():
        return {"thesis": THESIS_PATH.read_text(encoding="utf-8")}
    return {"thesis": ""}


@router.post("/")
def save_thesis(payload: dict[str, Any]) -> dict[str, str]:
    thesis = str(payload.get("thesis", ""))
    THESIS_PATH.write_text(thesis, encoding="utf-8")
    return {"status": "saved", "thesis": thesis}
