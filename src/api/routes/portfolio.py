from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.db.schema import SessionLocal
from src.db import queries


router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class PortfolioKPICreate(BaseModel):
    company_id: int
    metric_name: str
    value: float
    period: str | None = None


class PortfolioKPIUpdate(BaseModel):
    metric_name: str | None = None
    value: float | None = None
    period: str | None = None


def _kpi_to_dict(kpi: Any) -> dict[str, Any]:
    return {
        "id": kpi.id,
        "company_id": kpi.company_id,
        "metric_name": kpi.metric_name,
        "value": kpi.value,
        "period": kpi.period,
        "recorded_at": kpi.recorded_at.isoformat() if kpi.recorded_at else None,
    }


@router.get("/")
def list_portfolio_kpis(
    company_id: int | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    kpis = queries.list_portfolio_kpis(db, company_id=company_id, limit=limit, offset=offset)
    return {"items": [_kpi_to_dict(k) for k in kpis], "count": len(kpis)}


@router.get("/{kpi_id}")
def get_portfolio_kpi(kpi_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    kpi = queries.get_portfolio_kpi(db, kpi_id)
    if kpi is None:
        raise HTTPException(status_code=404, detail="Portfolio KPI not found")
    return _kpi_to_dict(kpi)


@router.post("/", status_code=201)
def create_portfolio_kpi(
    payload: PortfolioKPICreate, db: Session = Depends(get_db)
) -> dict[str, Any]:
    kpi = queries.create_portfolio_kpi(db, **payload.model_dump(exclude_none=True))
    return _kpi_to_dict(kpi)


@router.patch("/{kpi_id}")
def update_portfolio_kpi(
    kpi_id: int, payload: PortfolioKPIUpdate, db: Session = Depends(get_db)
) -> dict[str, Any]:
    kpi = queries.update_portfolio_kpi(
        db, kpi_id, **payload.model_dump(exclude_none=True)
    )
    if kpi is None:
        raise HTTPException(status_code=404, detail="Portfolio KPI not found")
    return _kpi_to_dict(kpi)


@router.delete("/{kpi_id}")
def delete_portfolio_kpi(kpi_id: int, db: Session = Depends(get_db)) -> Response:
    deleted = queries.delete_portfolio_kpi(db, kpi_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Portfolio KPI not found")
    return Response(status_code=204)
