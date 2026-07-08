"""CLI for TradingAgents.

    uv run python -m tradingagents decide AAPL [--date 2026-06-15] [--config ...]
    uv run python -m tradingagents eval [--config configs/tradingagents.yaml]

`decide` runs the full agent pipeline for one (ticker, session) and prints the
final decision plus where the transcript was written. `eval` runs the cached
point-in-time evaluation loop over the config window.
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from core import lake, paths
from tradingagents.config import load_ta_config
from tradingagents.evaluate import _cache_decision, run_eval
from tradingagents.graph import TradingAgentsGraph
from tradingagents.llm import OllamaLLM
from tradingagents.memory import DecisionMemory


def build_graph(cfg) -> TradingAgentsGraph:
    quick = OllamaLLM(cfg.quick_model, cfg.temperature, cfg.llm_timeout)
    deep = OllamaLLM(cfg.deep_model, cfg.temperature, cfg.llm_timeout)
    memory = DecisionMemory(horizon_sessions=cfg.horizon_sessions)
    return TradingAgentsGraph(cfg, quick, deep, memory=memory)


def cmd_decide(args: argparse.Namespace) -> None:
    cfg = load_ta_config(args.config)
    graph = build_graph(cfg)
    panel = lake.read_curated_prices()
    macro = lake.read_curated_macro()
    fundamentals = events = None
    if (paths.CURATED / "fundamentals.parquet").exists():
        fundamentals = pd.read_parquet(paths.CURATED / "fundamentals.parquet")
    if (paths.CURATED / "filing_events.parquet").exists():
        events = pd.read_parquet(paths.CURATED / "filing_events.parquet")
    asof = pd.Timestamp(args.date) if args.date else panel["date"].max()
    graph.memory.update_outcomes(panel, benchmark=cfg.benchmark)
    decision = graph.propagate(
        args.ticker.upper(), asof, panel, macro=macro, fundamentals=fundamentals, events=events
    )
    _cache_decision(decision)
    print(json.dumps({k: v for k, v in decision.to_dict().items() if k != "transcript"}, indent=2))
    print(f"transcript: artifacts/tradingagents/cache/{decision.ticker}_{decision.asof}.json")


def cmd_eval(args: argparse.Namespace) -> None:
    cfg = load_ta_config(args.config)
    out = run_eval(cfg, build_graph(cfg))
    keys = ("window", "book_equal_weight", "spy_buy_and_hold")
    print(json.dumps({k: out[k] for k in keys}, indent=2))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="tradingagents")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_decide = sub.add_parser("decide", help="one full agent-pipeline decision")
    p_decide.add_argument("ticker")
    p_decide.add_argument("--date", default=None, help="session date (default: last in lake)")
    p_decide.add_argument("--config", default=None)
    p_decide.set_defaults(func=cmd_decide)

    p_eval = sub.add_parser("eval", help="cached PIT evaluation over the config window")
    p_eval.add_argument("--config", default=None)
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
