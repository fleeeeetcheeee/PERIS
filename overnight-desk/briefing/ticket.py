"""Morning trade-ticket generation. The system's FINAL output — a human-readable
markdown ticket the user executes manually on Chase Self-Directed. No execution
code exists anywhere in this repo, by hard constraint.

Honest-reporting rules: no language implying guaranteed/expected profit; gate and
fallback status stated plainly; suppressed entirely if data is stale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

from core.config import Config
from ledger import db as ledger_db
from llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "> Research decision-support output, not investment advice. Signals are model "
    "rankings with substantial uncertainty; losses are possible and expected in some "
    "periods. Execute manually only if you accept the risks."
)

BRIEF_SYSTEM = (
    "You are a buy-side research assistant writing a short morning note. Use ONLY the "
    "numbers given to you — never invent, recompute, or extrapolate figures. Never imply "
    "guaranteed or expected profit. Two short paragraphs, plain language."
)


@dataclass
class TicketLine:
    ticker: str
    score: float
    rank: int
    weight: float
    shares: int
    limit_price: str  # decimal string
    last_close: float
    action: str  # buy / sell / hold


def to_lines(
    targets: pd.Series,
    previous: pd.Series,
    scores: pd.Series,
    closes: pd.Series,
    capital: float,
) -> list[TicketLine]:
    lines: list[TicketLine] = []
    all_names = targets.index.union(previous.index)
    ranked = scores.reindex(all_names).fillna(scores.min() if len(scores) else 0)
    order = ranked.rank(ascending=False).astype(int)
    for ticker in sorted(all_names, key=lambda t: order.get(t, 999)):
        tgt_w = float(targets.get(ticker, 0.0))
        prev_w = float(previous.get(ticker, 0.0))
        close = closes.get(ticker)
        if close is None or pd.isna(close):
            continue
        delta_w = tgt_w - prev_w
        shares = int(abs(delta_w) * capital / close)
        if shares == 0 and abs(delta_w) > 0:
            continue  # position too small for one share at this capital
        action = "hold" if abs(delta_w) < 1e-9 else ("buy" if delta_w > 0 else "sell")
        if action == "hold" and tgt_w == 0:
            continue
        # Limit prices: +/- 20 bps around last close, stored as decimal strings.
        buffer = Decimal("1.002") if action == "buy" else Decimal("0.998")
        limit = (Decimal(str(close)) * buffer).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        lines.append(
            TicketLine(
                ticker=ticker,
                score=float(scores.get(ticker, float("nan"))),
                rank=int(order.get(ticker, 0)),
                weight=tgt_w,
                shares=shares if action != "hold" else 0,
                limit_price=str(limit),
                last_close=float(close),
                action=action,
            )
        )
    return lines


def render_markdown(
    asof: str,
    lines: list[TicketLine],
    gate_notes: list[str],
    fallback_mode: bool,
    fallback_reason: str,
    capital: float,
    mode: str,
) -> str:
    hdr = [
        f"# Overnight Desk — Trade Ticket for {asof}",
        "",
        f"Mode: **{mode}**"
        + ("  |  **FALLBACK MODE** — " + fallback_reason if fallback_mode else ""),
        f"Sizing basis: ${capital:,.0f} (paper)",
        "",
    ]
    if gate_notes:
        hdr += ["## Gates", *[f"- {n}" for n in gate_notes], ""]

    rows = [
        "| # | Action | Ticker | Shares | Limit | Last close | Target wt | Score |",
        "|---|--------|--------|--------|-------|------------|-----------|-------|",
    ]
    for ln in lines:
        rows.append(
            f"| {ln.rank} | {ln.action.upper()} | {ln.ticker} | {ln.shares or '—'} | "
            f"${ln.limit_price} | ${ln.last_close:,.2f} | {ln.weight:.1%} | {ln.score:+.4f} |"
        )
    if not lines:
        rows = ["*No trades today — targets unchanged within the minimum-trade band.*"]

    return "\n".join(hdr + rows + ["", DISCLAIMER, ""])


def llm_commentary(asof: str, lines: list[TicketLine], gate_notes: list[str]) -> str | None:
    client = OllamaClient()
    if not client.available():
        return None
    facts = (
        "\n".join(
            f"- {ln.action.upper()} {ln.ticker}: {ln.shares} shares, limit ${ln.limit_price}, "
            f"target weight {ln.weight:.1%}, model score {ln.score:+.4f}"
            for ln in lines
        )
        or "- no trades today"
    )
    gates = "\n".join(f"- {g}" for g in gate_notes) or "- none"
    prompt = (
        f"Morning note for {asof}. Facts (use only these):\n{facts}\n\nActive gates:\n{gates}\n\n"
        "Explain in two short paragraphs what the ticket does and which risk gates are active."
    )
    return client.generate(BRIEF_SYSTEM, prompt)


def log_to_ledger(asof: str, lines: list[TicketLine], fallback_mode: bool) -> int:
    rows = [
        {
            "ticket_date": asof,
            "ticker": ln.ticker,
            "score": ln.score,
            "rank": ln.rank,
            "weight": ln.weight,
            "shares": ln.shares,
            "limit_price": ln.limit_price,
            "side": ln.action,
            "rationale": f"model rank {ln.rank}",
            "fallback_mode": int(fallback_mode),
        }
        for ln in lines
        if ln.action in ("buy", "sell")
    ]
    n = ledger_db.record_signals(rows)
    logger.info("ledger: %d signals recorded as pending for %s", n, asof)
    return n


def resolve_mode(cfg: Config) -> str:
    """Paper-trading gate: `live` stays disabled until the ledger shows >= 3 months
    of tracked signals. Do not remove or shortcut this gate."""
    status = ledger_db.paper_gate_status()
    if status["live_allowed"]:
        return "paper"  # live must also be explicitly requested; default stays paper
    return "paper"
