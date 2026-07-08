"""Point-in-time evaluation loop for the TradingAgents strategy.

EXPLORATORY by design: an LLM decision costs ~a dozen model calls, so this runs
one decision every `cadence_sessions` per ticker over a short window — nothing
like the walk-forward harness. Results are NOT comparable to the baseline
strategy's headline and are never a promotion input; the honesty rules still
apply to any claim made from them (report the window, compare against B&H on
the SAME window, no cherry-picking).

Accounting: a decision at session t's close sets the target weight from t+1
onward (no same-close execution); costs = |weight change| * cost_bps_per_side.
Decisions are cached per (ticker, session) in artifacts/tradingagents/cache/
so reruns and crashes never re-spend LLM calls.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from core import lake, paths
from tradingagents.config import TradingAgentsConfig
from tradingagents.graph import Decision, TradingAgentsGraph

logger = logging.getLogger(__name__)

CACHE_DIR = paths.ARTIFACTS / "tradingagents" / "cache"


def _cached_decision(ticker: str, asof: pd.Timestamp) -> Decision | None:
    p = CACHE_DIR / f"{ticker}_{asof.date()}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return Decision(**d)


def _cache_decision(decision: Decision) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / f"{decision.ticker}_{decision.asof}.json"
    p.write_text(json.dumps(decision.to_dict(), indent=2))


def _metrics(returns: pd.Series) -> dict:
    if len(returns) == 0:
        return {}
    equity = (1 + returns).cumprod()
    dd = float((equity / equity.cummax() - 1).min())
    vol = float(returns.std(ddof=1) * np.sqrt(252))
    return {
        "cumulative_return": float(equity.iloc[-1] - 1),
        "ann_return": float(returns.mean() * 252),
        "ann_vol": vol,
        "sharpe": float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
        if returns.std(ddof=1) > 0
        else 0.0,
        "max_drawdown": dd,
        "n_days": int(len(returns)),
    }


def run_eval(cfg: TradingAgentsConfig, graph: TradingAgentsGraph) -> dict:
    panel = lake.read_curated_prices()
    macro = lake.read_curated_macro()
    fundamentals = events = None
    fpath = paths.CURATED / "fundamentals.parquet"
    epath = paths.CURATED / "filing_events.parquet"
    if fpath.exists():
        fundamentals = pd.read_parquet(fpath)
    if epath.exists():
        events = pd.read_parquet(epath)

    close = panel.pivot(index="date", columns="ticker", values="close").sort_index()
    sessions = close.index
    start = pd.Timestamp(cfg.eval_start)
    end = pd.Timestamp(cfg.eval_end) if cfg.eval_end else sessions[-1]
    window = sessions[(sessions >= start) & (sessions <= end)]
    decision_days = window[:: cfg.cadence_sessions]
    logger.info(
        "eval window %s..%s: %d decision days x %d tickers",
        window[0].date(),
        window[-1].date(),
        len(decision_days),
        len(cfg.tickers),
    )

    per_ticker: dict[str, dict] = {}
    strat_returns: dict[str, pd.Series] = {}
    for ticker in cfg.tickers:
        ret = close[ticker].pct_change().reindex(window).fillna(0.0)
        weights = pd.Series(0.0, index=window)
        decisions: list[Decision] = []
        prev_w = 0.0
        for day in decision_days:
            dec = _cached_decision(ticker, day)
            if dec is None:
                if graph.memory is not None:
                    graph.memory.update_outcomes(panel, benchmark=cfg.benchmark)
                dec = graph.propagate(
                    ticker,
                    day,
                    panel,
                    macro=macro,
                    fundamentals=fundamentals,
                    events=events,
                    prev_weight=prev_w,
                )
                _cache_decision(dec)
            elif graph.memory is not None:
                graph.memory.record(dec)  # idempotent; keeps memory in sync with cache
            decisions.append(dec)
            # effective from the NEXT session through the next decision day
            weights.loc[weights.index > day] = dec.target_weight
            prev_w = dec.target_weight
        turnover = weights.diff().abs().fillna(weights.iloc[0] if len(weights) else 0.0)
        costs = turnover * cfg.cost_bps_per_side / 1e4
        net = weights * ret - costs
        strat_returns[ticker] = net
        actions = pd.Series([d.action for d in decisions])
        per_ticker[ticker] = {
            "strategy": _metrics(net),
            "buy_and_hold": _metrics(ret),
            "decisions": {
                "n": len(decisions),
                "buy": int((actions == "BUY").sum()),
                "hold": int((actions == "HOLD").sum()),
                "sell": int((actions == "SELL").sum()),
                "rejected_by_pm": int(sum(d.verdict == "REJECT" for d in decisions)),
            },
        }

    book = pd.DataFrame(strat_returns).mean(axis=1)  # equal-weight per-ticker books
    bench_ret = close[cfg.benchmark].pct_change().reindex(window).fillna(0.0)
    out = {
        "window": {"start": str(window[0].date()), "end": str(window[-1].date())},
        "config": {
            "tickers": cfg.tickers,
            "cadence_sessions": cfg.cadence_sessions,
            "debate_rounds": cfg.debate_rounds,
            "risk_rounds": cfg.risk_rounds,
            "quick_model": getattr(graph.quick, "model", "?"),
            "deep_model": getattr(graph.deep, "model", "?"),
        },
        "per_ticker": per_ticker,
        "book_equal_weight": _metrics(book),
        "spy_buy_and_hold": _metrics(bench_ret),
        "note": "EXPLORATORY: short-window LLM eval; not comparable to the walk-forward headline",
    }
    if graph.memory is not None:
        graph.memory.update_outcomes(panel, benchmark=cfg.benchmark)
    path = paths.ARTIFACTS / "tradingagents" / f"eval_{window[0].date()}_{window[-1].date()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", path)
    return out
