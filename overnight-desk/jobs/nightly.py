"""Nightly orchestrator (cron entrypoint, ~7:00 PM ET after US close).

    uv run python -m jobs.nightly --stage all --date 2026-07-02
    uv run python -m jobs.nightly --stage reconcile --fills fills.csv

Stages are idempotent and resumable; each writes its artifact under
data/curated/nightly/<date>/. Any failure exits non-zero and logs loudly — a
briefing is never emitted from stale data (explicit "no ticket today" instead).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from briefing import ticket as ticket_mod
from core import lake, paths
from core.calendar import last_completed_session, next_sessions
from core.config import Config, load_config
from features.build import FEATURES_CURATED, build_matrix, feature_columns
from features.macro_regime import _pivot_lagged
from ingestion.finnhub_client import FinnhubClient
from ingestion.run import ingest_macro, ingest_prices
from ledger import db as ledger_db
from models import predict
from portfolio import construct
from portfolio.gates import (
    GateDecision,
    drawdown_gate,
    earnings_gate,
    jump_regime_gate,
    vix_term_structure_gate,
)

logger = logging.getLogger("nightly")

STAGES = ["ingest", "features", "score", "construct", "brief"]


def stage_dir(asof: str) -> Path:
    d = paths.CURATED / "nightly" / asof
    d.mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------------------------------------------- stages


def stage_ingest(cfg: Config, asof: str) -> None:
    prices = ingest_prices(cfg)
    ingest_macro(cfg)
    last = str(prices["date"].max().date())
    if last < asof:
        raise RuntimeError(f"ingest finished but curated data ends {last} < asof {asof}")


def stage_features(cfg: Config, asof: str) -> None:
    # Pin the panel to the config's universe: the lake may hold extra tickers from
    # experiment backfills (wave 5 curated ~700 names), and features like spillover
    # depend on panel composition — production must not drift with lake contents.
    from core.universe import load_universe

    tickers = [m.ticker for m in load_universe(cfg.universe_file)]
    panel = lake.read_curated_prices(tickers=tickers)
    macro = lake.read_curated_macro()
    matrix = build_matrix(panel, macro, with_labels=True)
    matrix.to_parquet(FEATURES_CURATED, index=False)
    logger.info("features: %s", str(matrix.shape))


def stage_score(cfg: Config, asof: str) -> None:
    matrix = pd.read_parquet(FEATURES_CURATED)
    matrix["date"] = pd.to_datetime(matrix["date"])
    today = matrix[matrix["date"] == pd.Timestamp(asof)].reset_index(drop=True)
    if today.empty:
        raise RuntimeError(f"no feature rows for {asof} — run features stage first")
    cols = feature_columns(matrix)
    result = predict.score(today, cols)
    out = today[["date", "ticker"]].copy()
    out["raw_score"] = result.scores
    out["score"] = _smooth_scores(cfg, asof, out.set_index("ticker")["raw_score"]).values
    out.to_parquet(stage_dir(asof) / "scores.parquet", index=False)
    (stage_dir(asof) / "score_meta.json").write_text(
        json.dumps({"fallback_mode": result.fallback_mode, "reason": result.reason})
    )
    logger.info("score: %d names (%s)", len(out), result.reason)


def _smooth_scores(cfg: Config, asof: str, raw: pd.Series) -> pd.Series:
    """EMA of raw scores over recent sessions (matches the backtest's smoothing).

    Uses stored raw_score history from prior nightly runs; truncating to ~5 halflives
    leaves <3% of EWM weight behind. Falls back to raw scores when history is absent
    (first runs) — identical to what the backtest's min_periods=1 does.
    """
    hl = cfg.portfolio.score_smoothing_halflife
    if hl <= 0:
        return raw
    root = paths.CURATED / "nightly"
    history: list[pd.Series] = []
    if root.exists():
        prior = sorted(d.name for d in root.iterdir() if d.is_dir() and d.name < asof)
        for name in prior[-(5 * hl) :]:
            f = root / name / "scores.parquet"
            if f.exists():
                df = pd.read_parquet(f)
                col = "raw_score" if "raw_score" in df.columns else "score"
                history.append(df.set_index("ticker")[col].rename(name))
    frame = pd.concat([*history, raw.rename(asof)], axis=1).T
    return frame.ewm(halflife=hl, min_periods=1).mean().iloc[-1].reindex(raw.index)


def _previous_targets(asof: str) -> pd.Series:
    prev = _previous_targets_dated(asof)
    return prev[1] if prev else pd.Series(dtype=float)


def _previous_targets_dated(asof: str) -> tuple[str, pd.Series] | None:
    full = _previous_targets_full(asof)
    if full is None:
        return None
    name, data = full
    return name, pd.Series(data["weights"], dtype=float)


def _previous_targets_full(asof: str) -> tuple[str, dict] | None:
    root = paths.CURATED / "nightly"
    if not root.exists():
        return None
    prior = sorted(d.name for d in root.iterdir() if d.is_dir() and d.name < asof)
    for name in reversed(prior):
        f = root / name / "targets.json"
        if f.exists():
            return name, json.loads(f.read_text())
    return None


def _sessions_between(start: str, end: str) -> int:
    """Trading sessions strictly after `start` up to and including `end`."""
    from core.calendar import sessions

    s = sessions(pd.Timestamp(start).date(), pd.Timestamp(end).date())
    return max(0, len(s) - 1)


def _mark_paper_equity(cfg: Config, asof: str, targets: pd.Series, prices: pd.DataFrame) -> None:
    """Roll the paper equity mark forward so the drawdown breaker tracks reality.

    equity(asof) = equity(prev) * (1 + return of prev targets prev->asof) - trade costs
    on today's weight changes (stock bps as the conservative rate). This mark feeds the
    circuit breaker only — reported performance always comes from the backtester.
    """
    prev = _previous_targets_dated(asof)
    latest = ledger_db.latest_equity()
    if latest is None:
        ledger_db.mark_equity(asof, cfg.portfolio.capital)
        return
    if prev is None or prev[0] >= asof:
        return
    prev_date, prev_w = prev
    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    t0, t1 = pd.Timestamp(prev_date), pd.Timestamp(asof)
    if t0 not in close.index or t1 not in close.index or prev_w.empty:
        port_ret = 0.0
    else:
        rel = (close.loc[t1] / close.loc[t0] - 1).reindex(prev_w.index).fillna(0.0)
        port_ret = float((prev_w * rel).sum())
    all_names = targets.index.union(prev_w.index)
    turnover = float(
        (targets.reindex(all_names, fill_value=0.0) - prev_w.reindex(all_names, fill_value=0.0))
        .abs()
        .sum()
    )
    cost = turnover * cfg.costs.stock_bps_per_side / 1e4
    equity_new = latest[0] * (1 + port_ret - cost)
    ledger_db.mark_equity(asof, equity_new)
    logger.info(
        "paper equity: %.2f -> %.2f (ret %.3f%%, turnover cost %.3f%%)",
        latest[0],
        equity_new,
        port_ret * 100,
        cost * 100,
    )


def stage_construct(cfg: Config, asof: str) -> None:
    sdir = stage_dir(asof)

    # Rebalance cadence: between rebalances, carry previous weights and emit no trades
    # (mirrors the backtest's drift behavior exactly).
    prev_full = _previous_targets_full(asof)
    if prev_full is not None and cfg.portfolio.rebalance_every > 1:
        prev_name, prev_data = prev_full
        last_rebalance = prev_data.get("last_rebalance", prev_name)
        elapsed = _sessions_between(last_rebalance, asof)
        if elapsed < cfg.portfolio.rebalance_every:
            note = (
                f"no rebalance today (cadence: {elapsed}/{cfg.portfolio.rebalance_every} "
                f"sessions since {last_rebalance})"
            )
            (sdir / "targets.json").write_text(
                json.dumps(
                    {
                        "asof": asof,
                        "weights": prev_data["weights"],
                        "gate_notes": [note],
                        "exposure_scale": prev_data.get("exposure_scale", 1.0),
                        "allow_new_entries": prev_data.get("allow_new_entries", True),
                        "last_rebalance": last_rebalance,
                    },
                    indent=2,
                )
            )
            logger.info("construct: %s", note)
            return

    scores_df = pd.read_parquet(sdir / "scores.parquet")
    scores = scores_df.set_index("ticker")["score"]

    prices = lake.read_curated_prices()
    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    rets = close.pct_change()
    ts = pd.Timestamp(asof)
    vol63 = (rets.rolling(63, min_periods=40).std() * (252**0.5)).loc[ts]
    trailing = rets.loc[:ts].tail(cfg.portfolio.vol_lookback_days)

    decision = GateDecision()
    macro = lake.read_curated_macro()
    vix = vix3m = None
    if macro is not None:
        wide = _pivot_lagged(macro)
        row = wide.loc[wide.index <= ts]
        if not row.empty:
            vix = row["VIX"].iloc[-1] if "VIX" in row else None
            vix3m = row["VIX3M"].iloc[-1] if "VIX3M" in row else None
    decision = vix_term_structure_gate(decision, vix, vix3m, cfg.gates)

    if cfg.gates.jump_model:
        from models.regime import regime_series

        bench = close[cfg.benchmark].pct_change().dropna()
        states = regime_series(bench.loc[:ts], penalty=cfg.gates.jump_penalty)
        state = int(states.iloc[-1]) if len(states) else None
        decision = jump_regime_gate(decision, state, cfg.gates)

    finnhub = FinnhubClient()
    earnings = None
    if finnhub.available:
        horizon = next_sessions(ts.date(), cfg.gates.earnings_exclusion_sessions + 2)
        earnings = finnhub.earnings_calendar(asof, str(horizon[-1].date()))
    decision = earnings_gate(
        decision,
        earnings,
        ts,
        list(next_sessions(ts.date(), cfg.gates.earnings_exclusion_sessions)),
        cfg.gates,
    )

    eq = ledger_db.latest_equity()
    if eq:
        decision = drawdown_gate(decision, eq[0], eq[1], cfg.gates)

    eligible = pd.Series(True, index=scores.index)
    eligible &= vol63.reindex(scores.index).notna()
    eligible &= ~pd.Index(scores.index).isin(decision.excluded)

    previous = _previous_targets(asof)
    targets = construct.build_targets(
        scores=scores.dropna(),
        vol=vol63.reindex(scores.index),
        eligible=eligible,
        trailing_returns=trailing,
        previous=previous,
        cfg=cfg.portfolio,
        exposure_scale=decision.exposure_scale,
        allow_new_entries=decision.allow_new_entries,
    )
    (sdir / "targets.json").write_text(
        json.dumps(
            {
                "asof": asof,
                "weights": {k: round(float(v), 6) for k, v in targets.items()},
                "gate_notes": decision.notes,
                "exposure_scale": decision.exposure_scale,
                "allow_new_entries": decision.allow_new_entries,
                "last_rebalance": asof,
            },
            indent=2,
        )
    )
    logger.info("construct: %d positions, gates: %s", len(targets), decision.notes or "none")


def stage_brief(cfg: Config, asof: str) -> None:
    paths.BRIEFINGS.mkdir(parents=True, exist_ok=True)
    out_path = paths.BRIEFINGS / f"ticket_{asof}.md"

    # Stale-data suppression: curated must include the asof session.
    prices = lake.read_curated_prices()
    last = str(prices["date"].max().date())
    if last < asof:
        out_path.write_text(
            f"# Overnight Desk — {asof}\n\n**NO TICKET TODAY** — curated data ends {last}, "
            f"older than the last trading day. Fix ingest and re-run.\n"
        )
        logger.error("briefing SUPPRESSED: stale data (%s < %s)", last, asof)
        sys.exit(1)

    sdir = stage_dir(asof)
    scores_df = pd.read_parquet(sdir / "scores.parquet")
    scores = scores_df.set_index("ticker")["score"]
    tgt = json.loads((sdir / "targets.json").read_text())
    score_meta = json.loads((sdir / "score_meta.json").read_text())
    targets = pd.Series(tgt["weights"], dtype=float)
    previous = _previous_targets(asof)
    closes = prices[prices["date"] == pd.Timestamp(asof)].set_index("ticker")["close"]

    lines = ticket_mod.to_lines(targets, previous, scores, closes, cfg.portfolio.capital)
    mode = ticket_mod.resolve_mode(cfg)
    md = ticket_mod.render_markdown(
        asof,
        lines,
        tgt["gate_notes"],
        score_meta["fallback_mode"],
        score_meta["reason"],
        cfg.portfolio.capital,
        mode,
    )
    prose = ticket_mod.llm_commentary(asof, lines, tgt["gate_notes"])
    if prose:
        md += f"\n## Commentary\n\n{prose}\n"
    gate = ledger_db.paper_gate_status()
    md += f"\n---\nPaper-trading gate: {gate['reason']}\n"
    out_path.write_text(md)

    ticket_mod.log_to_ledger(asof, lines, score_meta["fallback_mode"])
    _mark_paper_equity(cfg, asof, targets, prices)
    logger.info("briefing written: %s", out_path)


def stage_reconcile(cfg: Config, asof: str, fills_csv: str | None) -> None:
    """Morning, manual: record actual fills, compare realized vs modeled slippage."""
    if not fills_csv:
        raise SystemExit("reconcile needs --fills fills.csv (ticket_date,ticker,shares,price)")
    fills = pd.read_csv(fills_csv, dtype={"price": str})
    n = 0
    with ledger_db.connect() as con:
        for _, f in fills.iterrows():
            row = con.execute(
                "SELECT id, limit_price FROM signals WHERE ticket_date=? AND ticker=?",
                (f["ticket_date"], f["ticker"]),
            ).fetchone()
            if row is None:
                logger.warning("no signal for %s %s — skipped", f["ticket_date"], f["ticker"])
                continue
            con.execute(
                "INSERT INTO fills (signal_id, fill_date, shares, price, paper) VALUES (?,?,?,?,1)",
                (row["id"], f["ticket_date"], int(f["shares"]), str(f["price"])),
            )
            con.execute("UPDATE signals SET status='filled' WHERE id=?", (row["id"],))
            # realized slippage vs the modeled cost, in bps of the limit price
            limit = float(row["limit_price"])
            realized_bps = abs(float(f["price"]) - limit) / limit * 1e4
            month = str(f["ticket_date"])[:7]
            prev = con.execute(
                "SELECT realized_bps, n_fills FROM slippage WHERE month=?", (month,)
            ).fetchone()
            if prev and prev["n_fills"]:
                k = prev["n_fills"]
                avg = (prev["realized_bps"] * k + realized_bps) / (k + 1)
                con.execute(
                    "UPDATE slippage SET realized_bps=?, n_fills=? WHERE month=?",
                    (avg, k + 1, month),
                )
            else:
                con.execute(
                    "INSERT OR REPLACE INTO slippage (month, modeled_bps, realized_bps, n_fills)"
                    " VALUES (?,?,?,1)",
                    (month, cfg.costs.stock_bps_per_side, realized_bps),
                )
            n += 1
    logger.info("reconcile: %d fills recorded (slippage table updated)", n)


# ----------------------------------------------------------------- main


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=[*STAGES, "all", "reconcile"])
    parser.add_argument(
        "--date", default=None, help="session date YYYY-MM-DD (default: last completed)"
    )
    parser.add_argument("--config", default=str(paths.CONFIGS / "baseline.yaml"))
    parser.add_argument("--fills", default=None, help="fills CSV for reconcile stage")
    args = parser.parse_args()

    cfg = load_config(args.config)
    asof = args.date or str(last_completed_session().date())
    logger.info("nightly asof=%s stage=%s", asof, args.stage)

    if args.stage == "reconcile":
        stage_reconcile(cfg, asof, args.fills)
        return

    stages = STAGES if args.stage == "all" else [args.stage]
    fns = {
        "ingest": stage_ingest,
        "features": stage_features,
        "score": stage_score,
        "construct": stage_construct,
        "brief": stage_brief,
    }
    for name in stages:
        logger.info("=== stage: %s ===", name)
        try:
            fns[name](cfg, asof)
        except SystemExit:
            raise
        except Exception:
            logger.exception(
                "stage %s FAILED — aborting (no briefing from stale/partial data)", name
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
