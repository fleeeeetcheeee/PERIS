from __future__ import annotations

import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .schema import Company, PipelineStage, PortfolioKPI, Signal

# Strips trailing ticker / CIK parentheticals from EDGAR display names
# e.g. "QXO, Inc.  (QXO, QXO-PB)" → "QXO, Inc."
_TICKER_SUFFIX_RE = re.compile(r"\s*\([A-Z0-9,\-\s]{1,30}\)\s*$")


def _normalize_name(name: str) -> str:
    """Normalize a company name for deduplication comparison."""
    normalized = _TICKER_SUFFIX_RE.sub("", name).strip()
    # Also collapse extra whitespace and strip common legal suffixes variation
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.lower()


def _apply_updates(instance: Any, updates: dict[str, Any]) -> None:
    for field, value in updates.items():
        if hasattr(instance, field):
            setattr(instance, field, value)


def create_company(session: Session, **company_data: Any) -> Company:
    """Insert a company, returning the existing row if a same-name company already exists.

    Deduplication checks both exact name (case-insensitive) and a normalized
    form that strips trailing ticker/parenthetical suffixes from EDGAR names.
    """
    name = company_data.get("name", "")
    if name:
        # 1. Exact case-insensitive match
        existing = session.scalars(
            select(Company).where(Company.name.ilike(name)).limit(1)
        ).first()
        if existing is not None:
            return existing
        # 2. Normalized match — catches "QXO, Inc." vs "QXO, Inc.  (QXO, QXO-PB)"
        normalized = _normalize_name(name)
        all_candidates = session.scalars(
            select(Company).where(Company.name.ilike(f"{normalized.split()[0]}%")).limit(50)
        ).all()
        for candidate in all_candidates:
            if _normalize_name(candidate.name) == normalized:
                return candidate
    company = Company(**company_data)
    session.add(company)
    session.commit()
    session.refresh(company)
    return company


def get_company(session: Session, company_id: int) -> Company | None:
    return session.get(Company, company_id)


def get_top_companies(
    session: Session, limit: int = 20, min_score: float = 70.0
) -> list[Company]:
    """Return companies ordered by score descending, filtered by min_score."""
    statement = (
        select(Company)
        .where(Company.score >= min_score)
        .where(Company.name != "_MACRO_DATA_")
        .order_by(Company.score.desc())
        .limit(limit)
    )
    return list(session.scalars(statement))


def list_companies(session: Session, limit: int | None = None, offset: int = 0) -> list[Company]:
    statement = select(Company).order_by(Company.created_at.desc(), Company.id.desc()).offset(offset)
    if limit is not None:
        statement = statement.limit(limit)
    return list(session.scalars(statement))


def update_company(session: Session, company_id: int, **company_data: Any) -> Company | None:
    company = get_company(session, company_id)
    if company is None:
        return None

    _apply_updates(company, company_data)
    session.commit()
    session.refresh(company)
    return company


def delete_company(session: Session, company_id: int) -> bool:
    company = get_company(session, company_id)
    if company is None:
        return False

    session.delete(company)
    session.commit()
    return True


def create_pipeline_stage(session: Session, **stage_data: Any) -> PipelineStage:
    pipeline_stage = PipelineStage(**stage_data)
    session.add(pipeline_stage)
    session.commit()
    session.refresh(pipeline_stage)
    return pipeline_stage


def get_pipeline_stage(session: Session, pipeline_stage_id: int) -> PipelineStage | None:
    return session.get(PipelineStage, pipeline_stage_id)


def list_pipeline_stages(
    session: Session,
    company_id: int | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> list[PipelineStage]:
    statement = select(PipelineStage)
    if company_id is not None:
        statement = statement.where(PipelineStage.company_id == company_id)
    statement = statement.order_by(PipelineStage.updated_at.desc(), PipelineStage.id.desc()).offset(offset)
    if limit is not None:
        statement = statement.limit(limit)
    return list(session.scalars(statement))


def update_pipeline_stage(
    session: Session,
    pipeline_stage_id: int,
    **stage_data: Any,
) -> PipelineStage | None:
    pipeline_stage = get_pipeline_stage(session, pipeline_stage_id)
    if pipeline_stage is None:
        return None

    _apply_updates(pipeline_stage, stage_data)
    session.commit()
    session.refresh(pipeline_stage)
    return pipeline_stage


def delete_pipeline_stage(session: Session, pipeline_stage_id: int) -> bool:
    pipeline_stage = get_pipeline_stage(session, pipeline_stage_id)
    if pipeline_stage is None:
        return False

    session.delete(pipeline_stage)
    session.commit()
    return True


def create_portfolio_kpi(session: Session, **kpi_data: Any) -> PortfolioKPI:
    portfolio_kpi = PortfolioKPI(**kpi_data)
    session.add(portfolio_kpi)
    session.commit()
    session.refresh(portfolio_kpi)
    return portfolio_kpi


def get_portfolio_kpi(session: Session, portfolio_kpi_id: int) -> PortfolioKPI | None:
    return session.get(PortfolioKPI, portfolio_kpi_id)


def list_portfolio_kpis(
    session: Session,
    company_id: int | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> list[PortfolioKPI]:
    statement = select(PortfolioKPI)
    if company_id is not None:
        statement = statement.where(PortfolioKPI.company_id == company_id)
    statement = statement.order_by(PortfolioKPI.recorded_at.desc(), PortfolioKPI.id.desc()).offset(offset)
    if limit is not None:
        statement = statement.limit(limit)
    return list(session.scalars(statement))


def update_portfolio_kpi(
    session: Session,
    portfolio_kpi_id: int,
    **kpi_data: Any,
) -> PortfolioKPI | None:
    portfolio_kpi = get_portfolio_kpi(session, portfolio_kpi_id)
    if portfolio_kpi is None:
        return None

    _apply_updates(portfolio_kpi, kpi_data)
    session.commit()
    session.refresh(portfolio_kpi)
    return portfolio_kpi


def delete_portfolio_kpi(session: Session, portfolio_kpi_id: int) -> bool:
    portfolio_kpi = get_portfolio_kpi(session, portfolio_kpi_id)
    if portfolio_kpi is None:
        return False

    session.delete(portfolio_kpi)
    session.commit()
    return True


def create_signal(session: Session, **signal_data: Any) -> Signal:
    signal = Signal(**signal_data)
    session.add(signal)
    session.commit()
    session.refresh(signal)
    return signal


def get_signal(session: Session, signal_id: int) -> Signal | None:
    return session.get(Signal, signal_id)


def list_signals(
    session: Session,
    company_id: int | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> list[Signal]:
    statement = select(Signal)
    if company_id is not None:
        statement = statement.where(Signal.company_id == company_id)
    statement = statement.order_by(Signal.created_at.desc(), Signal.id.desc()).offset(offset)
    if limit is not None:
        statement = statement.limit(limit)
    return list(session.scalars(statement))


def update_signal(session: Session, signal_id: int, **signal_data: Any) -> Signal | None:
    signal = get_signal(session, signal_id)
    if signal is None:
        return None

    _apply_updates(signal, signal_data)
    session.commit()
    session.refresh(signal)
    return signal


def delete_signal(session: Session, signal_id: int) -> bool:
    signal = get_signal(session, signal_id)
    if signal is None:
        return False

    session.delete(signal)
    session.commit()
    return True
