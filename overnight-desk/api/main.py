"""FastAPI app (reused PERIS layout). Serves briefings, ledger, and performance to the
dashboard. Read-only by design — there is no order or execution endpoint, ever.

Run: uv run uvicorn api.main:app --port 8100
"""

from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import PaperGate, PerformanceSummary, Signal, Ticket
from core import paths
from ledger import db as ledger_db

app = FastAPI(title="Overnight Desk API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok", "service": "overnight-desk-api"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def _ticket_for(date: str) -> Ticket:
    md_path = paths.BRIEFINGS / f"ticket_{date}.md"
    if not md_path.exists():
        raise HTTPException(404, f"no ticket for {date}")
    markdown = md_path.read_text()
    signals = [
        Signal(**{**s, "fallback_mode": bool(s["fallback_mode"])})
        for s in ledger_db.signals_for_date(date)
    ]
    return Ticket(
        date=date,
        markdown=markdown,
        signals=signals,
        suppressed="NO TICKET TODAY" in markdown,
    )


@app.get("/ticket/latest", response_model=Ticket)
async def latest_ticket() -> Ticket:
    if not paths.BRIEFINGS.exists():
        raise HTTPException(404, "no briefings yet — run jobs.nightly")
    tickets = sorted(paths.BRIEFINGS.glob("ticket_*.md"))
    if not tickets:
        raise HTTPException(404, "no briefings yet — run jobs.nightly")
    date = tickets[-1].stem.removeprefix("ticket_")
    return _ticket_for(date)


@app.get("/ticket/{date}", response_model=Ticket)
async def ticket_by_date(date: str) -> Ticket:
    return _ticket_for(date)


@app.get("/signals/{date}", response_model=list[Signal])
async def signals(date: str) -> list[Signal]:
    return [
        Signal(**{**s, "fallback_mode": bool(s["fallback_mode"])})
        for s in ledger_db.signals_for_date(date)
    ]


@app.get("/gate/paper", response_model=PaperGate)
async def paper_gate() -> PaperGate:
    return PaperGate(
        **{k: v for k, v in ledger_db.paper_gate_status().items() if k in PaperGate.model_fields}
    )


@app.get("/performance", response_model=PerformanceSummary)
async def performance() -> PerformanceSummary:
    f = paths.ARTIFACTS / "backtest_baseline.json"
    if not f.exists():
        raise HTTPException(404, "no backtest summary — run python -m backtest.run")
    summary = json.loads(f.read_text())
    note = (
        "Walk-forward out-of-sample, net of modeled costs. "
        + (
            "UNDERPERFORMS SPY buy-and-hold on this window. "
            if summary.get("underperforms_spy")
            else ""
        )
        + (
            "Survivorship bias present (no point-in-time constituents file)."
            if summary.get("survivorship_bias_warning")
            else ""
        )
    )
    return PerformanceSummary(summary=summary, note=note.strip())
