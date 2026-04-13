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


class PipelineStageCreate(BaseModel):
    company_id: int
    stage: str
    owner: str | None = None
    notes: str | None = None


class PipelineStageUpdate(BaseModel):
    stage: str | None = None
    owner: str | None = None
    notes: str | None = None


def _stage_to_dict(ps: Any) -> dict[str, Any]:
    return {
        "id": ps.id,
        "company_id": ps.company_id,
        "stage": ps.stage,
        "owner": ps.owner,
        "notes": ps.notes,
        "updated_at": ps.updated_at.isoformat() if ps.updated_at else None,
    }


@router.get("/")
def list_pipeline_stages(
    company_id: int | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    stages = queries.list_pipeline_stages(db, company_id=company_id, limit=limit, offset=offset)
    return {"items": [_stage_to_dict(s) for s in stages], "count": len(stages)}


@router.get("/{stage_id}")
def get_pipeline_stage(stage_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    stage = queries.get_pipeline_stage(db, stage_id)
    if stage is None:
        raise HTTPException(status_code=404, detail="Pipeline stage not found")
    return _stage_to_dict(stage)


@router.post("/", status_code=201)
def create_pipeline_stage(
    payload: PipelineStageCreate, db: Session = Depends(get_db)
) -> dict[str, Any]:
    stage = queries.create_pipeline_stage(db, **payload.model_dump(exclude_none=True))
    return _stage_to_dict(stage)


@router.patch("/{stage_id}")
def update_pipeline_stage(
    stage_id: int, payload: PipelineStageUpdate, db: Session = Depends(get_db)
) -> dict[str, Any]:
    stage = queries.update_pipeline_stage(
        db, stage_id, **payload.model_dump(exclude_none=True)
    )
    if stage is None:
        raise HTTPException(status_code=404, detail="Pipeline stage not found")
    return _stage_to_dict(stage)


@router.delete("/{stage_id}")
def delete_pipeline_stage(stage_id: int, db: Session = Depends(get_db)) -> Response:
    deleted = queries.delete_pipeline_stage(db, stage_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Pipeline stage not found")
    return Response(status_code=204)
