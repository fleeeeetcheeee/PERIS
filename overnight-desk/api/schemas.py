"""Pydantic v2 schemas at the API boundary (PERIS structure, adapted)."""

from __future__ import annotations

from pydantic import BaseModel


class Signal(BaseModel):
    id: int
    ticket_date: str
    ticker: str
    score: float
    rank: int
    weight: float
    shares: int
    limit_price: str
    side: str
    status: str
    fallback_mode: bool


class Ticket(BaseModel):
    date: str
    markdown: str
    signals: list[Signal]
    suppressed: bool


class PaperGate(BaseModel):
    live_allowed: bool
    reason: str
    tracked_days: int


class PerformanceSummary(BaseModel):
    summary: dict
    note: str
