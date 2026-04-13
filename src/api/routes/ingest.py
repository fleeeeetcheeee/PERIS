from __future__ import annotations

import logging
import threading

from fastapi import APIRouter

router = APIRouter()
logger = logging.getLogger(__name__)

_running = False


def _run_all() -> None:
    global _running
    try:
        from src.ingestion.sec_edgar import ingest_sec_edgar
        from src.ingestion.fred import ingest_fred_macro

        logger.info("Ingest run started")
        ingest_sec_edgar()
        ingest_fred_macro()
        logger.info("Ingest run complete")
    except Exception as exc:
        logger.error("Ingest run failed: %s", exc)
    finally:
        _running = False


@router.post("/run")
def run_ingest() -> dict[str, str]:
    global _running
    if _running:
        return {"status": "already_running"}
    _running = True
    thread = threading.Thread(target=_run_all, daemon=True)
    thread.start()
    return {"status": "started"}


@router.get("/status")
def ingest_status() -> dict[str, bool]:
    return {"running": _running}
