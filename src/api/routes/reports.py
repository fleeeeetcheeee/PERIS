from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from src.db.schema import SessionLocal
from src.db import queries


router = APIRouter()

REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "./reports"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/")
def list_reports() -> dict[str, Any]:
    """List all generated PDF reports."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(REPORTS_DIR.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {
        "items": [
            {
                "filename": p.name,
                "size_bytes": p.stat().st_size,
                "path": str(p),
            }
            for p in pdfs
        ],
        "count": len(pdfs),
    }


@router.get("/{filename}")
def get_report(filename: str) -> FileResponse:
    """Download a report PDF by filename."""
    # Prevent path traversal
    safe_name = Path(filename).name
    pdf_path = REPORTS_DIR / safe_name
    if not pdf_path.exists() or pdf_path.suffix != ".pdf":
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(str(pdf_path), media_type="application/pdf", filename=safe_name)


@router.post("/generate", status_code=202)
def generate_report(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Trigger an on-demand weekly report generation."""
    companies = [
        {
            "id": c.id,
            "name": c.name,
            "sector": c.sector,
            "score": c.score,
            "source": c.source,
        }
        for c in queries.list_companies(db, limit=200)
    ]
    pipeline_stages = [
        {"stage": ps.stage, "company_id": ps.company_id}
        for ps in queries.list_pipeline_stages(db, limit=500)
    ]
    portfolio_kpis = [
        {"metric_name": k.metric_name, "value": k.value, "period": k.period}
        for k in queries.list_portfolio_kpis(db, limit=500)
    ]
    signals = [
        {"signal_type": s.signal_type, "summary": s.summary}
        for s in queries.list_signals(db, limit=100)
    ]

    from src.agents.reporting_agent import ReportingAgent
    agent = ReportingAgent()
    result = agent.run({
        "companies": companies,
        "pipeline_stages": pipeline_stages,
        "portfolio_kpis": portfolio_kpis,
        "signals": signals,
    })
    return {"status": "generated", **result}
