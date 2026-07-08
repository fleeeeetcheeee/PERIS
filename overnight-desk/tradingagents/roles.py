"""Agent role prompts and structured-output parsing.

Communication protocol per the paper: analysts and decision-makers emit
STRUCTURED reports (fixed sections, machine-parseable tags); the researcher and
risk debates are free natural language. Parsers are fail-safe: an unparseable
decision degrades to HOLD and is logged, never crashes mid-pipeline.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

GROUNDING = (
    "Use ONLY the figures provided in the briefing. Never invent, recompute, or "
    "extrapolate numbers. You may reason about what the given figures imply. "
    "Be concise: at most 180 words."
)

ANALYST_ROLES: dict[str, tuple[str, str]] = {
    # role -> (system persona, which snapshot block it reads)
    "technical": (
        "You are the technical analyst of a trading firm. You read price action, "
        "momentum, mean-reversion, volatility and volume signals.",
        "technical",
    ),
    "fundamental": (
        "You are the fundamentals analyst of a trading firm. You judge business "
        "quality, growth, profitability and valuation from SEC filings.",
        "fundamental",
    ),
    "news": (
        "You are the macro/news analyst of a trading firm. You judge how the "
        "macro environment (rates, credit, volatility regime) bears on the stock.",
        "macro",
    ),
    "sentiment": (
        "You are the market-sentiment analyst of a trading firm. You read the "
        "market's own mood: earnings reactions, relative strength, breadth of up "
        "days, and crowd positioning proxies.",
        "mood",
    ),
}

ANALYST_TEMPLATE = """Ticker: {ticker} | Session: {asof}

Briefing (all figures computed as of the session close, point-in-time):
{block}

Write your analyst report with exactly these sections:
SIGNAL: one of BULLISH / BEARISH / NEUTRAL
KEY POINTS: 2-4 bullet points grounded in the briefing figures
RISKS: 1-2 bullet points on what would invalidate your read"""

RESEARCHER_SYSTEM = {
    "bull": (
        "You are the BULL researcher. Argue the strongest honest case FOR buying, "
        "grounded in the analyst reports. Rebut the bear's latest points directly. " + GROUNDING
    ),
    "bear": (
        "You are the BEAR researcher. Argue the strongest honest case AGAINST "
        "buying (or for selling), grounded in the analyst reports. Rebut the "
        "bull's latest points directly. " + GROUNDING
    ),
}

RESEARCHER_TEMPLATE = """Ticker: {ticker} | Session: {asof}

Analyst reports:
{reports}

Debate so far:
{debate}

Give your next argument (round {round} of {rounds})."""

MANAGER_SYSTEM = (
    "You are the research manager. You judge the bull/bear debate on argument "
    "quality and evidence, not on who spoke last. " + GROUNDING
)

MANAGER_TEMPLATE = """Ticker: {ticker} | Session: {asof}

Analyst reports:
{reports}

Full debate:
{debate}

Deliver your verdict with exactly these sections:
STANCE: one of BULLISH / BEARISH / NEUTRAL
PLAN: 2-3 sentences of investment thesis for the trader
WEAKNESS: the single strongest point from the losing side"""

TRADER_SYSTEM = (
    "You are the trader. You turn the research manager's plan into one concrete "
    "trading decision for the next week. You are judged on risk-adjusted returns "
    "net of costs, so do not churn: only trade when the evidence is clear. " + GROUNDING
)

TRADER_TEMPLATE = """Ticker: {ticker} | Session: {asof}

Research manager's verdict:
{plan}

Current position in {ticker}: {position}

Respond with exactly these sections:
DECISION: one of BUY / HOLD / SELL
CONVICTION: one of LOW / MEDIUM / HIGH
RATIONALE: 2-3 sentences"""

RISK_SYSTEM = {
    "aggressive": (
        "You are the aggressive risk debater. You push for taking more of the "
        "proposed risk when the reward justifies it. " + GROUNDING
    ),
    "neutral": (
        "You are the neutral risk debater. You weigh both sides and flag what "
        "the others exaggerate. " + GROUNDING
    ),
    "conservative": (
        "You are the conservative risk debater. You protect capital: drawdown, "
        "volatility regime, concentration and timing risks come first. " + GROUNDING
    ),
}

RISK_TEMPLATE = """Ticker: {ticker} | Session: {asof}

Trader's proposal:
{proposal}

Market context:
{macro}

Technical context:
{technical}

Risk discussion so far:
{debate}

Give your risk assessment of the proposal (round {round} of {rounds})."""

PM_SYSTEM = (
    "You are the portfolio manager with final authority. You approve, adjust, or "
    "reject the trader's proposal after hearing the risk team. Lessons from your "
    "own past decisions are provided — weigh them. " + GROUNDING
)

PM_TEMPLATE = """Ticker: {ticker} | Session: {asof}

Trader's proposal:
{proposal}

Risk team discussion:
{debate}

Lessons from recent past decisions (realized outcomes included):
{lessons}

Respond with exactly these sections:
VERDICT: one of APPROVE / ADJUST / REJECT
FINAL DECISION: one of BUY / HOLD / SELL
FINAL CONVICTION: one of LOW / MEDIUM / HIGH
REASON: 1-2 sentences"""


def _tag(text: str, name: str, allowed: tuple[str, ...]) -> str | None:
    """Extract `NAME: VALUE` where VALUE starts with one of the allowed words."""
    m = re.search(rf"{name}\s*[:\-]\s*\**\s*([A-Z]+)", text, flags=re.IGNORECASE)
    if m and m.group(1).upper() in allowed:
        return m.group(1).upper()
    for word in allowed:  # fallback: first allowed word anywhere after the tag name
        pattern = rf"{name}[^A-Za-z]*.{{0,40}}\b{word}\b"
        if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            return word
    return None


def parse_signal(text: str) -> str:
    return _tag(text, "SIGNAL", ("BULLISH", "BEARISH", "NEUTRAL")) or "NEUTRAL"


def parse_stance(text: str) -> str:
    return _tag(text, "STANCE", ("BULLISH", "BEARISH", "NEUTRAL")) or "NEUTRAL"


def parse_decision(text: str, tag: str = "DECISION") -> str:
    d = _tag(text, tag, ("BUY", "HOLD", "SELL"))
    if d is None:
        logger.warning("unparseable %s — failing safe to HOLD: %.80s", tag, text)
        return "HOLD"
    return d


def parse_conviction(text: str, tag: str = "CONVICTION") -> str:
    return _tag(text, tag, ("LOW", "MEDIUM", "HIGH")) or "LOW"


def parse_verdict(text: str) -> str:
    return _tag(text, "VERDICT", ("APPROVE", "ADJUST", "REJECT")) or "ADJUST"
