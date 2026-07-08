# Overnight Desk

Daily cross-sectional signal engine for US large-cap stocks and ETFs. Produces a morning
**trade ticket** (tickers, integer share counts, limit prices) that you execute manually on
Chase Self-Directed Investing. Research + decision support only — **there is no execution
code in this repo, by hard constraint** (see `CLAUDE2.md` at the repo root).

Scaffolded from PERIS: the FastAPI layout, FRED/EDGAR clients, Ollama wrapper, and the
rule-based-fallback scoring pattern are ports/adaptations.

## Quick start

```bash
cd overnight-desk
uv sync --extra dev

# 1. Backfill prices + macro (keyless: yfinance fallback + FRED CSV)
uv run python -m ingestion.run --config configs/baseline.yaml

# 2. Build features, train the LightGBM ranker (purged walk-forward)
uv run python -m features.build
uv run python -m models.train configs/model.yaml

# 3. Backtest on out-of-sample predictions, net of costs
uv run python -m backtest.run configs/baseline.yaml

# 4. Nightly pipeline (cron ~7pm ET) — emits briefings/ticket_<date>.md
uv run python -m jobs.nightly --stage all

# 5. Serve to a dashboard
uv run uvicorn api.main:app --port 8100

# Tests (lookahead, known-value fixtures, backtest golden file)
uv run pytest tests/ -x
```

## API keys (all optional; the system runs keyless out of the box)

Keys live in `overnight-desk/.env` (created, currently empty — copy values in as you get
them). Real environment variables override `.env`. Verify any time with:

```bash
uv run python -m ingestion.check_keys
```

| Env var           | Unlocks                                              | Sign up at                                  |
|-------------------|------------------------------------------------------|---------------------------------------------|
| `FINNHUB_API_KEY` | Earnings-date exclusion gate goes active             | finnhub.io/dashboard                         |
| `TIINGO_API_KEY`  | Tiingo becomes the primary EOD source (div-adjusted) | tiingo.com/account/api/token                 |
| `FRED_API_KEY`    | Keyed FRED API (otherwise keyless fredgraph CSV)     | fred.stlouisfed.org/docs/api/api_key.html    |

Without keys: prices come from the yfinance fallback (Stooq now fronts its CSV endpoint
with a browser-verification challenge, so the documented Stooq path is auto-skipped when
unreachable), and the earnings gate logs itself INACTIVE on every ticket.

**Tiingo free-tier reality:** the hourly request cap (~30–50/hr, observed empirically)
is below our 71-ticker universe, so a single-shot nightly ingest cannot complete on
Tiingo. Ingest tries sources in priority order (Tiingo → Stooq → yfinance) and, on a
429 storm, abandons the source *for the whole run* and falls through — each curation
uses exactly one vendor per run, because mixing adjusted series from two vendors inside
a ticker's history creates basis seams at every dividend. In practice yfinance remains
the nightly workhorse; the Finnhub earnings gate is unaffected (it needs ~1 call/night).

## Nightly schedule (launchd)

`~/Library/LaunchAgents/com.overnightdesk.nightly.plist` runs
`jobs/run_nightly.sh` weekdays at 7:10 PM ET; logs land in `logs/nightly_<date>.log`
(last 30 kept). Ollama runs as a brew service (`brew services start ollama`), so
commentary survives reboots.

**One-time step (required):** macOS privacy protection (TCC) blocks launchd jobs from
reading `~/Documents`, where this repo lives. Grant access once:
System Settings → Privacy & Security → **Full Disk Access** → add **/bin/zsh**
(⌘⇧G in the file picker to type the path) and toggle it on. Test with:

```bash
launchctl kickstart gui/$(id -u)/com.overnightdesk.nightly
tail -f logs/nightly_$(date +%Y-%m-%d).log
```

Until that grant, the agent fires on schedule but exits 127 ("Operation not permitted").
Alternative if you'd rather not grant zsh disk access: move the project out of
`~/Documents` (e.g. `~/projects/PERIS`) — TCC only guards Documents/Desktop/Downloads —
and update the two absolute paths in the plist and `jobs/run_nightly.sh`.

Each nightly `brief` stage also rolls a paper equity mark forward in the ledger
(prev targets' return net of an approximate turnover cost) so the drawdown circuit
breaker tracks the paper book; reported performance still comes only from the backtester.

## Layout

```
core/        config (pydantic-validated YAML), NYSE calendar, Parquet/DuckDB lake, universe
ingestion/   tiingo, stooq, yahoo (fallback), finnhub, fred, edgar clients + raw→curated validation
features/    one module per family: reversal, residual_momentum, trend, volume_liquidity,
             macro_regime, calendar_features — each `compute(panel, macro)` with declared LOOKBACK
models/      purged walk-forward harness, LightGBM training CLI, artifact registry, fallback scorer
portfolio/   top-K + inverse-vol + cap + vol-target construction; earnings/VIX/drawdown gates
backtest/    daily engine (costs always on), metrics (deflated Sharpe, hit-rate CI, vs-SPY)
briefing/    trade ticket rendering; Ollama writes prose only — never numbers
jobs/        nightly.py staged orchestrator (ingest→features→score→construct→brief, + reconcile)
ledger/      SQLite: signals, fills, equity marks, slippage; paper-trading gate lives here
api/         read-only FastAPI for the dashboard
tradingagents/  multi-agent LLM trading framework (arXiv:2412.20138) — see below
data/        Parquet lake (gitignored): raw/ immutable, curated/ validated, ledger.db
```

## TradingAgents (multi-agent LLM framework)

Implementation of Xiao, Sun, Luo & Wang, *TradingAgents: Multi-Agents LLM
Financial Trading Framework* (arXiv:2412.20138), adapted to this repo's rules:

```
uv run python -m tradingagents decide AAPL              # one full pipeline decision
uv run python -m tradingagents decide AAPL --date 2026-06-15
uv run python -m tradingagents eval                     # cached PIT eval over configs/tradingagents.yaml
```

Pipeline per decision: 4 analysts (technical / fundamentals / macro-news /
market-sentiment) write structured reports from a **point-in-time snapshot**
(prices ≤ t, macro < t, fundamentals filed < t, earnings reactions ≤ t — all
figures computed in pandas, the LLM never produces numbers) → bull vs bear
researchers debate (`debate_rounds`) → research manager verdict → trader
proposes BUY/HOLD/SELL + conviction → aggressive/conservative/neutral risk
debate (`risk_rounds`) → portfolio manager approves/adjusts/rejects with
lessons from its own realized past decisions (`artifacts/tradingagents/
decisions.jsonl`) injected. Deviations from the paper, forced by keylessness:
the social-sentiment analyst reads market-mood proxies (announcement-day
abnormal returns, relative strength, up-day breadth) instead of social feeds,
and both quick-think/deep-think roles default to the installed Ollama model.
Decisions map to weights in code (BUY×conviction → 0.33/0.66/1.0, long-only);
unparseable output fails safe to HOLD; a PM REJECT forces HOLD. Every decision
is cached (`artifacts/tradingagents/cache/`) with its full transcript, so eval
reruns never re-spend LLM calls. **The eval is exploratory**: a short-window,
few-ticker loop vs buy-and-hold on the same window — not the walk-forward
harness, never a promotion input, and not comparable to the headline numbers.

### The Desk (pixel UI + always-on background service)

```
zsh jobs/install_tradingagents_desk.sh   # installs both pieces below (idempotent)
```

1. **launchd agent `com.overnightdesk.tradingagents`** — runs
   `tradingagents.app` (FastAPI on :8102) permanently: RunAtLoad + KeepAlive,
   same zsh-entrypoint pattern as the nightly. Its `DeskWorker` thread wakes
   every 30 min, fills realized outcomes into the decision memory, and makes
   (or replays from cache) each ticker's decision for the latest session — so
   after every nightly ingest the floor does one real round of work and then
   idles for free. Logs: `logs/tradingagents_desk.log`.
2. **`~/Applications/TradingAgents Desk.app`** — pixel-art icon, openable from
   Spotlight; makes sure the server is up (kickstarts the agent if not) and
   opens the floor.

The floor (`tradingagents/ui/index.html`, self-contained, no CDN) shows every
agent as a pixel character at a desk — analyst row, research pit (the bull has
horns), trading, risk desk, PM corner office. Characters type with lit monitors
and speech bubbles while their stage runs (SSE from `/events`), doze when the
worker sleeps; click any character to read their latest output. Sidebar:
decision board (BUY/HOLD/SELL badges per ticker), realized track record from
the decision memory, and a live log. `/state` replays recent history so a
freshly opened window reconstructs the floor.

**Live watch mode + BUY/SELL notifications.** During NYSE hours the worker
switches to a `watch_poll_minutes` (10 min) rhythm and pulls delayed quotes
keyless via yfinance — quotes are shown live on the decision board but are
never written to the lake (the lake stays the official nightly record). When a
watchlist name moves ≥ `watch_move_pct` (2.5%) vs yesterday's close, the desk
runs a full agent REVISION with the intraday update injected as a
clearly-labelled unofficial block on top of the point-in-time snapshot (max one
revision per ticker per `revision_cooldown_minutes`). Whenever the desk's final
stance on a ticker becomes BUY or SELL — from the daily cycle or a revision — a
native macOS notification fires (title, ticker, conviction, PM verdict, target
weight, trigger) and the UI logs an alert. Repeat identical stances stay silent
unless `notify_all: true`. All signals are the model's research output for the
paper book, not investment advice.

Uninstall: `launchctl bootout gui/$UID/com.overnightdesk.tradingagents`,
delete the plist from `~/Library/LaunchAgents` and the app from
`~/Applications`.

## Guardrails baked in

- **No lookahead**: every feature family has a mechanical lookahead test (corrupt the
  future, values in the past must be bit-identical). Macro series are lagged one business
  day. Labels are the only forward-looking column and the CV purges 5 sessions + 2-day embargo.
- **Costs always on**: 7.5 bps/side stocks, 3 bps ETFs; gross returns are never a headline.
- **Honest reporting**: summaries carry deflated Sharpe, hit-rate CI, vs-SPY-after-costs,
  and an explicit survivorship-bias warning until `data/reference/constituents_pit.csv` exists.
- **Paper gate**: `live` stays disabled until the ledger shows ≥ 3 months of tracked signals.
- **Stale-data suppression**: a briefing older than the last trading day becomes an explicit
  "NO TICKET TODAY" notice and a non-zero exit.

## Known limitations (deliberate, documented)

- `data/reference/constituents_pit.csv` (from the free fja05680/sp500 dataset, with
  FB→META and UTX→RTX rename windows merged) gives every universe stock verified index
  membership across the whole backtest window, and the membership mask is applied in
  `load_dataset`. Residual caveat: the universe was still *chosen* from today's
  mega-caps, so a selection-bias echo remains until ranking runs over the full
  historical index membership.
- yfinance-fallback closes are split+dividend adjusted; Stooq closes (if that path is used)
  are split-adjusted only — flagged in `ingestion/stooq_client.py`.
- The Next.js dashboard reuse from PERIS is not wired up yet; the API (`api/main.py`)
  exposes everything a dashboard needs (`/ticket/latest`, `/signals/{date}`, `/performance`,
  `/gate/paper`).
- **The strategy still underperforms SPY buy-and-hold after costs.** Two experiment
  waves so far (all on identical walk-forward OOS predictions, costs on):
  - Turnover wave (`backtest/experiment.py` → `artifacts/experiment_turnover.json`):
    smoothing (EMA halflife 3) + 5-session cadence took net from −3.0%/yr to +5.1%/yr.
  - Wave 1, model (`models/experiment_wave1.py` → `artifacts/experiment_wave1.json`):
    LambdaRank objective + fractional-differentiation + spillover features took net
    to +8.6%/yr, Sharpe 0.79 — deflated Sharpe 0.569 at the cumulative n_trials=17.
  - Wave 2, construction/gates (`backtest/experiment_wave2.py` →
    `artifacts/experiment_wave2.json`): **negative result, nothing promoted.** HRP and
    RMT-cleaned weighting cost 4–6 Sharpe points vs inverse-vol at K=12; the
    statistical jump-model regime gate correctly identified stress regimes (18.7% of
    days) but the strategy was already *profitable* during them (+3.1 bps/day) —
    smoothing + cadence + inverse-vol pre-empt the gate — so gating only forfeited
    return and worsened path-dependent drawdown (−17.8% → −23.0%). All three
    mechanisms remain in the codebase behind config flags (defaults off).
  - Wave 3, meta-labeling (`backtest/experiment_wave3.py` →
    `artifacts/experiment_wave3.json`): **negative result, nothing promoted** —
    despite the best point estimates so far. A secondary walk-forward classifier
    (`models/meta.py`) predicting P(pick pays off), used for calibrated bet sizing
    (`meta_mode: sized`), showed +9.2%/yr and Sharpe 0.87 vs the incumbent's 0.79.
    But the improvement fails every honest test: the Sharpe difference (+0.08) has
    block-bootstrap P(≤0) = 0.31; the daily return difference is p = 0.70; and the
    mechanism decomposition shows the meta model's cuts were *indiscriminate* — the
    shaved positions earned the same per unit weight as the kept book (−0.9 bps/d,
    p = 0.79), so the Sharpe bump is an incidental exposure-path effect, not skill.
    The model's own calibration curve agrees (payoff by prob quintile 52.2% → 56.2%
    around a 54.1% base rate). `meta_mode: tilt|gate|sized` remains available in
    `PortfolioConfig` (default `off`).
  - Wave 4, new signal channels (`models/experiment_wave4.py` →
    `artifacts/experiment_wave4.json`): **negative result, nothing promoted.**
    Path-signature Lévy areas (`features/signature.py`: momentum timing,
    price/volume lead-lag) HURT outright (Sharpe 0.79 → 0.72). Turbulence
    conditioners (`features/turbulence.py`: Kritzman-Li Mahalanobis + MST H0
    persistence) posted the best point estimate of any wave (+9.6%/yr, Sharpe 0.85)
    but failed every honest test: daily diff p = 0.63, Sharpe-difference bootstrap
    P(≤0) = 0.37, paired IC diff p = 0.96 — and the gain does NOT concentrate on
    high-turbulence days (p = 0.72), which is the only mechanism a regime
    conditioner has. Both families stay in the codebase under
    `features.build.EXPERIMENTAL_FAMILIES`, deliberately OUTSIDE the promoted
    `FAMILIES` list: the promotion gate is a point-estimate rule, so leaving
    rejected features in the default matrix would let a routine retrain promote
    exactly the luck this wave rejected.
  - Wave 5, universe expansion (`models/experiment_wave5.py` →
    `artifacts/experiment_wave5.json`): **negative result, nothing promoted — and
    the wave that caught the project's most dangerous false positive.** The
    universe was expanded to the full point-in-time S&P 500 (665 stocks via
    `ingestion/build_sp500_universe.py`: 24 verified rename mappings, membership
    windows coalesced; 596 curated, 93% member-day coverage, 52% for departed
    names — the residue is reported, not hidden). Findings:
    1. Breadth destroys the signal: Sharpe 0.79 → 0.16, and IC collapses to
       +0.002 (negative even within the original 71 names). Mechanism: ranking
       RAW returns across a heterogeneous-vol universe turns lambdarank into a
       volatility bet — 83% of picks landed on names with median 47% ann vol.
    2. The principled repair (rank of vol-adjusted forward return) recovers only
       Sharpe 0.29 on the broad universe. Expansion is dead: the edge does not
       transfer beyond the mega-cap pool.
    3. The same vol-adjusted label on the 71-name universe printed **Sharpe 1.03
       vs incumbent 0.79 — entirely a window artifact.** The label's 40-session
       vol warm-up shifts the walk-forward's first OOS block ~2 months, past the
       COVID crash (incumbent: −14.1% in exactly those 40 days). On the common
       window: incumbent 1.07, voladj 1.04. **Rule: cross-variant comparisons are
       valid only on the common OOS window; experiment artifacts now record each
       variant's window.**
    This wave also fixed real pipeline bugs: curation no longer drops tickers for
    real crashes/earnings gaps (only split-ratio-shaped moves and garbage series
    fail — the old rule was injecting survivorship bias by excluding AAL, OXY,
    PCG, SBNY...), and experiments refuse to run when another config's ingest has
    re-curated the shared lake mid-flight (the 19:10 nightly did exactly that).
  - Wave-5 aftermath — **macro-vintage sensitivity** (see
    `../Research/2026-07-06_macro-vintage-sensitivity.md`): two same-day headline
    runs differing only in when the FRED macro table was fetched scored Sharpe
    0.791 vs 0.684. Proven by elimination (deterministic reproduction; zero
    backtest-path divergence on fixed predictions; fold-1 bit-identical; first
    prediction divergence exactly at the 2023-10-16 fold boundary; PIT masks and
    price lake ruled out byte-level). A no-`mac_*` probe (trial #28) scores 0.613 —
    the macro features' marginal value (+0.07) is the same size as the vintage
    noise (0.107). **The honest headline is therefore a range: Sharpe ≈ 0.68–0.79
    net, deflated Sharpe 0.39–0.51 at n_trials=28.** Ingest now archives a dated
    macro snapshot in `data/raw/fred/` on every run so vintages are
    reconstructable.
  - Wave 6, first NON-PRICE data (`models/experiment_wave6.py` →
    `artifacts/experiment_wave6.json`): **negative result, nothing promoted —
    sixth straight negative wave.** New keyless data pipeline: SEC EDGAR
    companyfacts fundamentals (`ingestion/fundamentals.py` → 48.8k fact vintages,
    each row carrying its own `filed` date) and filing events (10.9k rows with
    acceptance timestamps). Features: `features/fundamentals.py` (7 ratios/growths,
    as-of joined strictly AFTER `filed`; valuation yields use dei
    `EntityPublicFloat` because adjusted close × as-reported shares is wrong by
    every future split factor) and `features/pead.py` (estimate-free PEAD:
    surprise = announcement-day abnormal return vs SPY at the 8-K 2.02 reaction
    session). All five variants lost to the incumbent (0.684) on the IDENTICAL
    OOS window: fund 0.648 (p=0.96), pead 0.635 (p=0.77), fund+pead 0.581
    (p=0.65), pead_v2 0.635 (p=0.82), fund+pead_v2 0.438 (p=0.30); paired t and
    21d-block-bootstrap stats are now computed inside the experiment and stored
    in the JSON. Univariate daily ICs corroborate: best new feature
    `fund_ni_growth` t=1.83 across 10 tested — noise. Notable data bug caught by
    cadence stats, not by tests: the 10-Q fallback rule created 794 spurious
    announcement events (routine 10-Qs filed weeks after the real 8-K reset the
    drift clock; 2787 → 1993 events after fix; median announcements/ticker-year
    4.0 as expected) — the v2 variants re-ran PEAD on the fixed stream and were
    counted as trials 32–33. TTM construction quirk worth remembering: cash-flow
    statements report YTD only, so quarterly CFO comes from same-start YTD
    differencing; four consecutive quarter ENDS span ~273 days (a 330+ day
    "TTM window" means a quarter is missing — reject, don't sum). Both families
    stay in `EXPERIMENTAL_FAMILIES`. Interpretation: on 71 mega-caps at a 5-day
    horizon, slow fundamental signals and announcement drift add nothing over
    price features — consistent with PEAD being weakest in large caps. Headline
    unchanged: **Sharpe 0.68–0.79 net, dSR now 0.35–0.51 at n_trials=33.**
  SPY did +14.9%/yr on the window. See `../Research/` for the wave-1 finding:
  the LambdaRank improvement is invisible (indeed negative) under mean rank IC.
  Consequently the artifact-promotion gate in `models/train.py` now uses walk-forward
  net Sharpe through the portfolio + costs (mean IC only as fallback for old
  artifacts). Any further change must follow the definition-of-done in CLAUDE2.md:
  walk-forward results after costs vs. incumbent, turnover delta, deflated Sharpe,
  updated golden files.
