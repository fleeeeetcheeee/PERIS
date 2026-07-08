# CLAUDE.md — Overnight Desk

Daily cross-sectional signal engine for US large-cap stocks and ETFs. Generates a morning
trade ticket (tickers, share counts, limit prices, rationale) that the user executes
**manually** on Chase Self-Directed Investing. This is a research + decision-support system,
not an automated trader.

## Hard constraints (never violate)

1. **No trade execution, ever.** Chase Self-Directed has no API and prohibits algorithmic
   orders. The system's final output is a human-readable trade ticket. Never add broker
   integrations, order-routing code, or anything that places trades.
2. **No lookahead bias.** Every feature at time `t` may only use data available at the close
   of day `t`. All joins must be point-in-time. Corporate actions, index membership, and
   fundamentals must be lagged to their public-availability date. If unsure whether a field
   is point-in-time safe, flag it in a comment and exclude it.
3. **Costs are always on.** Backtests and portfolio construction must include the transaction
   cost model (default 7.5 bps per side on stocks, 3 bps on ETFs) and turnover penalties.
   Never present or log gross-of-cost performance as a headline number.
4. **Purged walk-forward CV only.** No random k-fold on time series. Use expanding-window
   walk-forward with a purge gap ≥ the label horizon (5 trading days) plus a 2-day embargo.
5. **Paper-trading gate.** The `live` briefing mode stays disabled until the paper ledger
   shows ≥ 3 months of tracked signals. Do not remove or shortcut this gate.
6. **Honest reporting.** Briefings and dashboards must never use language implying guaranteed
   or expected profit. Report deflated Sharpe, hit rate with confidence intervals, and
   performance vs. SPY buy-and-hold after costs. If the strategy underperforms SPY, say so.
7. **Long-only, cash account assumptions.** No shorting, no margin, no options logic.

## Relationship to PERIS

This repo is scaffolded from PERIS (Private Equity Research Intelligence System). Reuse,
don't rewrite:

- `ingestion/edgar_client.py` and `ingestion/fred_client.py` — port as-is, adapt schemas.
- FastAPI app structure (`api/`) — same layout: routers, services, pydantic schemas.
- SQLite patterns — keep for metadata/ledger, but **market data lives in Parquet + DuckDB**,
  not SQLite. Do not store OHLCV rows in SQLite.
- Ollama integration (`llm/`) — reuse the client wrapper; the local model only writes
  briefing prose. **The LLM never produces numbers.** All figures in a briefing are computed
  upstream and injected into the prompt; the LLM formats and explains only.
- Next.js dashboard shell — reuse layout, auth-less local mode, and chart components.
- The rule-based fallback pattern from PERIS scoring applies here too: if the ML model is
  unavailable or fails validation checks, fall back to the rule-based reversal+momentum
  composite, and mark the briefing as "fallback mode".

## Architecture

```
overnight-desk/
├── data/                  # Parquet lake (gitignored)
│   ├── raw/               # As-pulled vendor data, immutable
│   ├── curated/           # Adjusted, validated OHLCV + features
│   └── ledger.db          # SQLite: signals, paper fills, real fills, reconciliation
├── ingestion/             # Tiingo, Stooq, Finnhub, FRED, EDGAR clients
├── features/              # Feature definitions (one module per family)
├── models/                # LightGBM ranker, training, walk-forward harness
├── portfolio/             # Selection, weighting, vol targeting, risk gates
├── backtest/              # Vectorized backtester + cost model
├── briefing/              # Trade ticket generation (Ollama)
├── api/                   # FastAPI (reused PERIS structure)
├── dashboard/             # Next.js (reused PERIS shell)
├── jobs/                  # nightly.py orchestrator (cron entrypoint)
└── tests/
```

## Nightly pipeline (jobs/nightly.py)

Runs after US close (schedule ~7:00 PM ET). Stages are idempotent and resumable:

1. `ingest` — pull EOD prices, corporate actions, macro series. Validate row counts and
   adjusted-price continuity before promoting raw → curated.
2. `features` — recompute feature matrix for the active universe.
3. `score` — load current model artifact, rank universe, apply regime filter.
4. `construct` — top-K (default 12) selection, inverse-vol weights, 10% position cap,
   portfolio vol target 12% annualized, turnover penalty.
5. `gates` — earnings-date exclusion (skip names reporting within 2 sessions), VIX term
   structure filter, max-drawdown circuit breaker (halt new entries at −10% from ledger HWM).
6. `brief` — render trade ticket (markdown + dashboard), log to ledger as `pending`.
7. `reconcile` — (morning, manual trigger) user records actual fills; compute realized
   slippage vs. cost model and update the slippage estimate monthly.

Any stage failure must fail loudly (non-zero exit, logged) — never silently skip and emit a
briefing from stale data. A briefing built on data older than the last trading day must be
suppressed with an explicit "no ticket today" notice.

## Universe & strategy

- Universe: current S&P 500 constituents + ~30 liquid sector/factor ETFs. Membership must be
  point-in-time in backtests (use a historical constituents file; do not backtest today's
  list into the past — that's survivorship bias).
- Label: next-5-day cross-sectional relative return (rank-normalized).
- Model: LightGBM ranker (LambdaRank or regression-on-ranks). Tree models are the default;
  do not introduce deep nets without a benchmarked reason.
- Feature families: short-term reversal, residual momentum, vol-scaled trend, volume/
  liquidity, macro regime (FRED), calendar. One module per family in `features/`, each
  exposing `compute(df) -> DataFrame` with declared lookback and point-in-time contract.
- Retrain monthly, expanding window, artifact versioned with training-data hash. A new model
  is promoted only if it beats the incumbent out-of-sample on the validation harness.

## Data sources & limits

| Source  | Use                        | Notes                                      |
|---------|----------------------------|--------------------------------------------|
| Tiingo  | Primary EOD prices         | Free tier; respect rate limits; cache all  |
| Stooq   | Bulk historical backfill   | One-time backfill into `data/raw/`         |
| Finnhub | Quotes, earnings calendar  | 60 calls/min free tier                     |
| FRED    | Macro series               | Reuse PERIS client                         |
| EDGAR   | Fundamentals (lagged)      | Reuse PERIS client; lag to filing date     |

yfinance is fallback only — never a primary dependency. All vendor pulls are cached; never
re-fetch data already in `data/raw/`.

## Conventions

- Python 3.12, `uv` for deps, `ruff` + `ruff format`, full type hints, pydantic v2 schemas
  at all module boundaries.
- Pandas with explicit `DatetimeIndex` in exchange timezone (America/New_York); no naive
  timestamps anywhere.
- Deterministic everything: seeds fixed in training, backtests reproducible from config.
  Configs are YAML in `configs/`, validated by pydantic; no magic numbers in code.
- Tests: every feature module gets a lookahead test (shuffle future rows → feature values at
  `t` must not change) and a known-value fixture test. Backtester has golden-file tests.
- Money math in float64 is fine for research, but the ledger stores share counts as int and
  prices as decimal strings.

## Commands

```bash
uv sync                                  # install
uv run python -m jobs.nightly --stage all --date 2026-07-02
uv run python -m backtest.run configs/baseline.yaml
uv run python -m models.train configs/model.yaml
uv run pytest tests/ -x
cd dashboard && npm run dev              # dashboard at localhost:3000
```

## Definition of done for any strategy change

A PR that touches features, model, or portfolio logic must include: (1) walk-forward results
after costs vs. incumbent, (2) turnover delta, (3) deflated Sharpe, (4) updated golden files.
"It looks better in-sample" is not evidence.
