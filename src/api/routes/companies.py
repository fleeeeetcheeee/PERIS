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


class CompanyCreate(BaseModel):
    name: str
    sector: str | None = None
    country: str | None = None
    employee_count: int | None = None
    revenue_estimate: float | None = None
    source: str | None = None
    score: float | None = None


class CompanyUpdate(BaseModel):
    name: str | None = None
    sector: str | None = None
    country: str | None = None
    employee_count: int | None = None
    revenue_estimate: float | None = None
    source: str | None = None
    score: float | None = None


def _company_to_dict(company: Any) -> dict[str, Any]:
    return {
        "id": company.id,
        "name": company.name,
        "sector": company.sector,
        "country": company.country,
        "employee_count": company.employee_count,
        "revenue_estimate": company.revenue_estimate,
        "source": company.source,
        "score": company.score,
        "created_at": company.created_at.isoformat() if company.created_at else None,
    }


@router.get("/")
def list_companies(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    companies = queries.list_companies(db, limit=limit, offset=offset)
    return {"items": [_company_to_dict(c) for c in companies], "count": len(companies)}


@router.get("/{company_id}")
def get_company(company_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    company = queries.get_company(db, company_id)
    if company is None:
        raise HTTPException(status_code=404, detail="Company not found")
    return _company_to_dict(company)


@router.post("/", status_code=201)
def create_company(
    payload: CompanyCreate, db: Session = Depends(get_db)
) -> dict[str, Any]:
    company = queries.create_company(db, **payload.model_dump(exclude_none=True))
    return _company_to_dict(company)


@router.patch("/{company_id}")
def update_company(
    company_id: int, payload: CompanyUpdate, db: Session = Depends(get_db)
) -> dict[str, Any]:
    company = queries.update_company(
        db, company_id, **payload.model_dump(exclude_none=True)
    )
    if company is None:
        raise HTTPException(status_code=404, detail="Company not found")
    return _company_to_dict(company)


@router.delete("/{company_id}")
def delete_company(company_id: int, db: Session = Depends(get_db)) -> Response:
    deleted = queries.delete_company(db, company_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Company not found")
    return Response(status_code=204)
