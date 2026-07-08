"""TradingAgents config — pydantic-validated YAML, same pattern as core.config."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class TradingAgentsConfig(BaseModel):
    tickers: list[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "NVDA"])
    benchmark: str = "SPY"

    # Debate depth. The paper's ablation favors small numbers; every extra round
    # is 2 (researchers) or 3 (risk) more LLM calls per decision.
    debate_rounds: int = 2
    risk_rounds: int = 1

    # None = auto-detect the first installed Ollama model. quick handles the
    # data-heavy analyst roles, deep the judgment roles (manager/trader/PM).
    quick_model: str | None = None
    deep_model: str | None = None
    temperature: float = 0.2
    llm_timeout: float = 300.0

    # Decision -> target weight (fraction of the per-ticker book, long-only).
    conviction_weights: dict[str, float] = Field(
        default_factory=lambda: {"LOW": 0.33, "MEDIUM": 0.66, "HIGH": 1.0}
    )

    # Evaluation loop: one decision every cadence_sessions per ticker.
    eval_start: date = date(2026, 1, 2)
    eval_end: date | None = None
    cadence_sessions: int = 5
    cost_bps_per_side: float = 7.5
    horizon_sessions: int = 5  # outcome horizon for the decision memory

    memory_lessons: int = 3  # recent outcomes injected into the PM prompt

    # Desk server (tradingagents.app): pixel-floor UI + SSE + background worker.
    server_port: int = 8102
    worker_poll_minutes: int = 30

    # Live watch mode: during NYSE hours the worker polls delayed quotes every
    # watch_poll_minutes; a move of >= watch_move_pct vs yesterday's close
    # triggers a fresh agent review with the intraday update injected (at most
    # one per ticker per revision_cooldown_minutes).
    watch_enabled: bool = True
    watch_poll_minutes: int = 10
    watch_move_pct: float = 2.5
    revision_cooldown_minutes: int = 90

    # macOS notifications when the desk's final action for a ticker becomes
    # BUY or SELL (notify_all: also when it repeats unchanged).
    notify: bool = True
    notify_all: bool = False


def load_ta_config(path: str | Path | None = None) -> TradingAgentsConfig:
    if path is None:
        from core import paths

        path = paths.CONFIGS / "tradingagents.yaml"
    p = Path(path)
    raw = yaml.safe_load(p.read_text()) if p.exists() else {}
    return TradingAgentsConfig.model_validate(raw or {})
