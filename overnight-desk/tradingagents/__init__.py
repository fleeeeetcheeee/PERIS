"""TradingAgents: multi-agent LLM trading framework.

Implementation of Xiao, Sun, Luo & Wang, "TradingAgents: Multi-Agents LLM
Financial Trading Framework" (arXiv:2412.20138), adapted to this repo's
constraints:

- ALL data comes from the existing keyless lake (prices, FRED macro, EDGAR
  fundamentals + filing events from wave 6) — no new vendors. The paper's
  social-sentiment analyst becomes a market-mood analyst fed by the market's
  own reactions (earnings-day abnormal returns, relative strength, volume),
  since keyless social feeds don't exist.
- THE LLM NEVER PRODUCES NUMBERS (repo rule): every figure the agents see is
  computed upstream in pandas and injected into prompts; agents output stances,
  arguments, and BUY/HOLD/SELL words which code maps to deterministic weights.
- Point-in-time: a decision for session t sees prices through t, macro through
  t-1 (FRED publication lag convention), fundamentals filed strictly before t,
  and announcement reactions on sessions <= t.
- Local-first LLMs: quick-think and deep-think roles both default to the
  installed Ollama model; either can be overridden per config.

Pipeline (one decision): 4 analysts -> bull/bear researcher debate ->
research manager verdict -> trader proposal -> risk debate (aggressive/
neutral/conservative) -> portfolio manager approval with decision memory.
"""

from tradingagents.config import TradingAgentsConfig, load_ta_config
from tradingagents.graph import TradingAgentsGraph

__all__ = ["TradingAgentsConfig", "TradingAgentsGraph", "load_ta_config"]
