"""Wave-5 experiment: expand the universe from 71 names to the full PIT S&P 500.

    uv run python -m models.experiment_wave5

Waves 2-4 exhausted construction, sizing, and price-derived features on the 71-name
snapshot; the remaining reliable multiplier on a fixed IC is BREADTH. This wave
ranks over every stock that held S&P 500 membership inside the window (~680
tickers, PIT-masked via constituents_pit.csv), with the promoted model and
portfolio config unchanged.

Survivorship honesty: departed members are fetched like any other ticker, but the
free vendor cannot serve every delisted name — the run reports member-day coverage
overall and for departed names specifically. Missing departed names bias results
UP (the missing names skew toward failures), so the coverage numbers are part of
the result, not a footnote.

Prereq: `uv run python -m ingestion.run --config configs/sp500.yaml` (fills the
lake; the curated table is restored to the baseline universe afterwards by
re-running baseline ingest — raw is immutable, so this is cheap).

Two variants:
- sp500_universe: the promoted config verbatim on the expanded panel.
- sp500_voladj_label: same, but the training label is the cross-sectional rank of
  the VOL-ADJUSTED forward return (fwd_ret / trailing 63d vol, both PIT). Diagnosis
  from the first variant: ranking raw returns across a heterogeneous-vol universe
  turns lambdarank into a volatility bet (83% of picks landed on 47%-vol names and
  IC collapsed even within the original 71). The vol-adjusted label removes that
  incentive; prediction targets risk-adjusted rank, the portfolio stays unchanged.

Third variant, baseline71_voladj_label: the vol-adjusted label on the PRODUCTION
71-name universe. Surfaced by accident (a nightly-ingest collision shrank the panel
mid-experiment and the "repair" variant trained on 71 names, printing Sharpe ~1.0);
run here deliberately, as its own counted trial. Mechanism hypothesis: even among
mega-caps, vol dispersion makes raw-return ranks partly a volatility bet.

Honest reporting: n_trials = 24 (waves 0-4) + 3 = 27.
Results -> artifacts/experiment_wave5.json.
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from backtest.engine import run_backtest
from backtest.metrics import summarize
from core import lake, paths
from core.config import Config, load_config
from core.universe import asset_type_map, load_universe
from features.build import build_matrix
from features.macro_regime import _pivot_lagged
from models.train import load_dataset, oos_predictions

logger = logging.getLogger(__name__)

N_TRIALS_TOTAL = 27
VARIANTS = ("sp500_universe", "sp500_voladj_label", "baseline71_voladj_label")
_CACHE_SUFFIX = {
    "sp500_universe": "sp500",
    "sp500_voladj_label": "sp500_voladj",
    "baseline71_voladj_label": "base71_voladj",
}


def _preds_cache(variant: str):
    return paths.ARTIFACTS / f"oos_preds_wave5_{_CACHE_SUFFIX[variant]}.parquet"


def _voladj_labels(matrix: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Replace `label` with the per-date rank of fwd_ret / trailing 63d vol.

    vol at t uses returns through t (trailing only); fwd_ret is the one
    deliberately forward-looking column, unchanged.
    """
    close = panel.pivot(index="date", columns="ticker", values="close").sort_index()
    vol = (close.pct_change().rolling(63, min_periods=40).std()).stack().rename("vol63")
    out = matrix.merge(vol.reset_index(), on=["date", "ticker"], how="left")
    adj = out["fwd_ret"] / out["vol63"].clip(lower=1e-4)
    pct = adj.groupby(out["date"]).rank(pct=True)
    out["label"] = pct - pct.groupby(out["date"]).transform("mean")
    out.loc[adj.isna(), "label"] = pd.NA
    return out.drop(columns=["vol63"])


def coverage_report(panel: pd.DataFrame, cfg: Config) -> dict:
    """Member-day coverage of the curated panel vs the PIT membership calendar."""
    pit = pd.read_csv(paths.REFERENCE / "constituents_pit.csv", parse_dates=["start", "end"])
    sessions = pd.DatetimeIndex(sorted(panel["date"].unique()))
    w0 = pd.Timestamp(cfg.start)
    have = panel.groupby("ticker")["date"].agg(["min", "max"])

    total_days = 0
    covered_days = 0
    departed_total = 0
    departed_covered = 0
    missing: list[str] = []
    for r in pit.itertuples():
        end = r.end if pd.notna(r.end) else sessions[-1]
        start = max(r.start, w0)
        if start > sessions[-1] or end < w0:
            continue
        n = int(((sessions >= start) & (sessions <= end)).sum())
        total_days += n
        departed = pd.notna(r.end)
        if departed:
            departed_total += n
        if r.ticker in have.index:
            lo, hi = have.loc[r.ticker, "min"], have.loc[r.ticker, "max"]
            c = int(((sessions >= max(start, lo)) & (sessions <= min(end, hi))).sum())
            covered_days += c
            if departed:
                departed_covered += c
        else:
            missing.append(r.ticker)
    return {
        "member_day_coverage": covered_days / total_days if total_days else 0.0,
        "departed_member_day_coverage": (
            departed_covered / departed_total if departed_total else 0.0
        ),
        "tickers_missing_entirely": sorted(set(missing)),
        "n_tickers_in_panel": int(panel["ticker"].nunique()),
    }


def run_wave5(cfg: Config) -> dict:
    members = load_universe(cfg.universe_file)
    panel = lake.read_curated_prices(tickers=[m.ticker for m in members])
    # The curated table is SHARED and any ingest run re-curates it to its own
    # config's universe (the 19:10 nightly re-curates to the 71-name production
    # set). Refuse to run on a shrunken panel instead of silently backtesting
    # cached broad-universe predictions against missing prices.
    if panel["ticker"].nunique() < len(members) / 2:
        raise RuntimeError(
            f"curated table holds {panel['ticker'].nunique()} of {len(members)} universe "
            f"tickers — another config re-curated the lake. Re-run: "
            f"uv run python -m ingestion.run --config configs/sp500.yaml"
        )
    cov = coverage_report(panel, cfg)
    logger.info(
        "coverage: %.1f%% of member-days (departed names: %.1f%%); %d tickers in panel, "
        "%d missing entirely",
        cov["member_day_coverage"] * 100,
        cov["departed_member_day_coverage"] * 100,
        cov["n_tickers_in_panel"],
        len(cov["tickers_missing_entirely"]),
    )

    macro = lake.read_curated_macro()
    matrix = None
    macro_vix = None
    if macro is not None:
        wide = _pivot_lagged(macro)
        macro_vix = wide[["VIX", "VIX3M"]].reset_index()
    bench = (
        panel[panel["ticker"] == cfg.benchmark]
        .set_index("date")["close"]
        .pct_change()
        .rename("bench")
    )

    base71 = [
        m.ticker for m in load_universe("data/reference/universe.csv")
    ]  # production universe for the baseline71 variant

    results: dict[str, dict] = {}
    daily_net: dict[str, pd.Series] = {}
    for variant in VARIANTS:
        vpanel = panel[panel["ticker"].isin(base71)] if variant.startswith("baseline71") else panel
        cache = _preds_cache(variant)
        if cache.exists():
            preds = pd.read_parquet(cache)
            preds["date"] = pd.to_datetime(preds["date"])
            logger.info("%s: using cached predictions (%d rows)", variant, len(preds))
        else:
            if variant.startswith("baseline71"):
                vm = _voladj_labels(build_matrix(vpanel, macro, with_labels=True), vpanel)
            else:
                if matrix is None:
                    logger.info(
                        "building feature matrix over %d tickers...", panel["ticker"].nunique()
                    )
                    matrix = build_matrix(panel, macro, with_labels=True)
                vm = matrix if variant == "sp500_universe" else _voladj_labels(matrix, panel)
            data, cols = load_dataset(cfg, matrix=vm)  # applies the PIT mask itself
            logger.info("%s: dataset %d rows, %d features", variant, len(data), len(cols))
            preds = oos_predictions(data, cols, cfg)
            preds.to_parquet(cache, index=False)

        bt = run_backtest(preds, vpanel, asset_type_map(members), cfg, macro_vix=macro_vix)
        summary = summarize(bt.net_returns, bench, bt.turnover, n_trials=N_TRIALS_TOTAL)
        # Label changes shift data availability and therefore the walk-forward's OOS
        # START — the voladj label's 40-session vol warm-up moved it past the COVID
        # crash and manufactured a fake Sharpe 1.03. Cross-variant comparisons are
        # only valid on the COMMON window; record each variant's window loudly.
        summary["oos_start"] = str(bt.net_returns.index.min().date())
        summary["oos_end"] = str(bt.net_returns.index.max().date())
        results[variant] = summary
        daily_net[variant] = bt.net_returns
        logger.info(
            "%-20s ann_net %+6.2f%%  sharpe %+5.2f  dSR %.3f  maxDD %6.1f%%  TO %4.1f%%",
            variant,
            summary["ann_return_net"] * 100,
            summary["sharpe_net"],
            summary["deflated_sharpe"],
            summary["max_drawdown"] * 100,
            summary["avg_daily_turnover"] * 100,
        )

    out = {"n_trials": N_TRIALS_TOTAL, "coverage": cov, "results": results}
    (paths.ARTIFACTS / "experiment_wave5.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(daily_net).to_parquet(paths.ARTIFACTS / "experiment_wave5_daily.parquet")
    logger.info("wrote artifacts/experiment_wave5.json")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_wave5(load_config(paths.CONFIGS / "sp500.yaml"))
