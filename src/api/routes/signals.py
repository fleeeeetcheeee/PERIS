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


class SignalCreate(BaseModel):
    company_id: int
    signal_type: str
    summary: str
    raw_data: dict[str, Any] | list[Any] | None = None
    confidence: float | None = None


class SignalUpdate(BaseModel):
    signal_type: str | None = None
    summary: str | None = None
    raw_data: dict[str, Any] | list[Any] | None = None
    confidence: float | None = None


def _signal_to_dict(signal: Any) -> dict[str, Any]:
    return {
        "id": signal.id,
        "company_id": signal.company_id,
        "signal_type": signal.signal_type,
        "summary": signal.summary,
        "raw_data": signal.raw_data,
        "confidence": signal.confidence,
        "created_at": signal.created_at.isoformat() if signal.created_at else None,
    }


@router.get("/")
def list_signals(
    company_id: int | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    signals = queries.list_signals(db, company_id=company_id, limit=limit, offset=offset)
    return {"items": [_signal_to_dict(s) for s in signals], "count": len(signals)}


@router.get("/{signal_id}")
def get_signal(signal_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    signal = queries.get_signal(db, signal_id)
    if signal is None:
        raise HTTPException(status_code=404, detail="Signal not found")
    return _signal_to_dict(signal)


@router.post("/", status_code=201)
def create_signal(
    payload: SignalCreate, db: Session = Depends(get_db)
) -> dict[str, Any]:
    signal = queries.create_signal(db, **payload.model_dump(exclude_none=True))
    return _signal_to_dict(signal)


@router.patch("/{signal_id}")
def update_signal(
    signal_id: int, payload: SignalUpdate, db: Session = Depends(get_db)
) -> dict[str, Any]:
    signal = queries.update_signal(
        db, signal_id, **payload.model_dump(exclude_none=True)
    )
    if signal is None:
        raise HTTPException(status_code=404, detail="Signal not found")
    return _signal_to_dict(signal)


@router.delete("/{signal_id}")
def delete_signal(signal_id: int, db: Session = Depends(get_db)) -> Response:
    deleted = queries.delete_signal(db, signal_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Signal not found")
    return Response(status_code=204)
