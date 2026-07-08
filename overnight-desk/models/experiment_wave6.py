"""Wave-6 experiment: EDGAR fundamentals + estimate-free PEAD features.

    uv run python -m models.experiment_wave6

Waves 2-5 exhausted construction, sizing, price-derived features, and breadth on
daily bars — five straight negatives put the edge ceiling near Sharpe 0.8 on the
71-name universe. This wave adds the first NON-PRICE data source: point-in-time
SEC fundamentals (features/fundamentals.py, as-of joined on `filed`) and
post-earnings-announcement drift proxied without estimates
(features/pead.py, announcement-day abnormal return from 8-K 2.02 timestamps).

Variants share ONE feature matrix (fund_/pead_ columns are NaN-tolerant, so data
availability — and therefore the OOS window — is identical across variants; the
JSON still records each window per the wave-5 rule):

- incumbent_rank_all : promoted feature set, unchanged
- rank_fund          : + fund_ (7 fundamental ratios/growths)
- rank_pead          : + pead_ (3 drift features)
- rank_fund_pead     : + both

Unlike waves 2-5, the paired honesty stats are computed HERE, not ad hoc after:
each challenger row carries the paired daily t-test vs the incumbent and the
21-day block bootstrap P(Sharpe diff <= 0), both on the COMMON OOS window.

Honest reporting: n_trials = 28 (waves 0-5 + probes) + 3 + 2 = 33. The +2 are the
_v2 PEAD variants: v1's event stream let routine 10-Qs filed weeks after the 8-K
count as fresh announcements (794 spurious clock-resets out of 2787 events, found
from event-rate stats); v2 reruns the pead variants on the fixed stream. The v1
rows stay in the JSON — cached predictions, counted trials.
Results -> artifacts/experiment_wave6.json (+ per-variant daily nets parquet).
"""

from __future__ import annotations

import copy
import json
import logging
import math

import numpy as np
import pandas as pd
from scipy import stats as sps

from backtest.engine import run_backtest
from backtest.metrics import summarize
from core import lake, paths
from core.config import Config, load_config
from core.universe import asset_type_map, load_universe
from features.macro_regime import _pivot_lagged
from models.train import load_dataset, oos_predictions
from models.walkforward import rank_ic

logger = logging.getLogger(__name__)

N_TRIALS_TOTAL = 33  # 28 (waves 0-5 incl. probes) + 3 (v1) + 2 (v2 pead event fix)

INCUMBENT_PREFIXES = ("rev_", "resmom_", "mom_", "trend_", "liq_", "mac_", "cal_", "ffd_", "spill_")
FUND_PREFIX = ("fund_",)
PEAD_PREFIX = ("pead_",)

VARIANTS: dict[str, tuple[str, ...]] = {
    "incumbent_rank_all": INCUMBENT_PREFIXES,
    "rank_fund": INCUMBENT_PREFIXES + FUND_PREFIX,
    "rank_pead": INCUMBENT_PREFIXES + PEAD_PREFIX,
    "rank_fund_pead": INCUMBENT_PREFIXES + FUND_PREFIX + PEAD_PREFIX,
    # v2 = same prefixes, matrix built AFTER the earnings_events fallback fix; the
    # v1 rows above resolve from cached predictions and keep their buggy stream
    "rank_pead_v2": INCUMBENT_PREFIXES + PEAD_PREFIX,
    "rank_fund_pead_v2": INCUMBENT_PREFIXES + FUND_PREFIX + PEAD_PREFIX,
}

BOOT_BLOCK = 21
BOOT_N = 2000


def _select_cols(all_cols: list[str], prefixes: tuple[str, ...]) -> list[str]:
    return [c for c in all_cols if c.startswith(prefixes)]


def _sharpe(x: np.ndarray) -> float:
    sd = x.std(ddof=1)
    return float(x.mean() / sd * math.sqrt(252)) if sd > 0 else 0.0


def paired_stats(challenger: pd.Series, incumbent: pd.Series, seed: int = 42) -> dict:
    """Paired daily t-test + block bootstrap of the Sharpe difference, on the
    common OOS window only (honesty rule #1)."""
    idx = challenger.index.intersection(incumbent.index)
    a = challenger.loc[idx].to_numpy()
    b = incumbent.loc[idx].to_numpy()
    diff = a - b
    t, p = sps.ttest_1samp(diff, 0.0)

    rng = np.random.default_rng(seed)
    n = len(diff)
    n_blocks = math.ceil(n / BOOT_BLOCK)
    sharpe_diffs = np.empty(BOOT_N)
    for i in range(BOOT_N):
        starts = rng.integers(0, n - BOOT_BLOCK + 1, n_blocks)
        sel = np.concatenate([np.arange(s, s + BOOT_BLOCK) for s in starts])[:n]
        sharpe_diffs[i] = _sharpe(a[sel]) - _sharpe(b[sel])
    return {
        "common_days": int(n),
        "common_start": str(idx.min().date()),
        "common_end": str(idx.max().date()),
        "mean_daily_diff_bps": float(diff.mean() * 1e4),
        "t_stat": float(t),
        "p_value": float(p),
        "sharpe_diff_point": _sharpe(a) - _sharpe(b),
        "bootstrap_p_sharpe_diff_le_0": float((sharpe_diffs <= 0).mean()),
    }


def event_coverage(data: pd.DataFrame) -> dict:
    """How much of the panel the new columns actually cover — thin coverage means
    any gain rides on a handful of names, which is a luck flag."""
    out = {}
    for col in ("fund_ey", "pead_surprise"):
        if col in data.columns:
            out[f"{col}_coverage"] = float(data[col].notna().mean())
    return out


def run_wave6(base_cfg: Config) -> dict:
    from features.build import FAMILIES, build_matrix, fundamentals, pead

    panel = lake.read_curated_prices()
    n_tickers = panel["ticker"].nunique()
    members = load_universe(base_cfg.universe_file)
    if n_tickers < len(members) - 5:
        raise RuntimeError(
            f"curated table holds {n_tickers} of {len(members)} universe tickers — "
            "another config re-curated the lake; re-run baseline ingest first"
        )
    macro = lake.read_curated_macro()
    matrix = build_matrix(panel, macro, with_labels=True, families=[*FAMILIES, fundamentals, pead])
    data, all_cols = load_dataset(base_cfg, matrix=matrix)
    logger.info("dataset: %d rows, %d total feature cols", len(data), len(all_cols))

    asset_types = asset_type_map(members)
    macro_vix = None
    if macro is not None:
        wide = _pivot_lagged(macro)
        macro_vix = wide[["VIX", "VIX3M"]].reset_index()
    bench = (
        panel[panel["ticker"] == base_cfg.benchmark]
        .set_index("date")["close"]
        .pct_change()
        .rename("bench")
    )

    results: dict[str, dict] = {}
    daily_net: dict[str, pd.Series] = {}
    for name, prefixes in VARIANTS.items():
        cfg = copy.deepcopy(base_cfg)
        cols = _select_cols(all_cols, prefixes)
        logger.info("=== %s: %d features ===", name, len(cols))

        cache = paths.ARTIFACTS / f"oos_preds_wave6_{name}.parquet"
        if cache.exists():
            preds = pd.read_parquet(cache)
            preds["date"] = pd.to_datetime(preds["date"])
            logger.info("%s: using cached predictions (%d rows)", name, len(preds))
        else:
            preds = oos_predictions(data, cols, cfg)
            preds.to_parquet(cache, index=False)

        joined = preds.merge(data[["date", "ticker", "label"]], on=["date", "ticker"])
        daily_ic = rank_ic(joined["score"], joined["label"], joined["date"])

        bt = run_backtest(preds, panel, asset_types, cfg, macro_vix=macro_vix)
        summary = summarize(bt.net_returns, bench, bt.turnover, n_trials=N_TRIALS_TOTAL)
        summary["oos_start"] = str(bt.net_returns.index.min().date())
        summary["oos_end"] = str(bt.net_returns.index.max().date())
        daily_net[name] = bt.net_returns

        results[name] = {
            "n_features": len(cols),
            "mean_ic": float(daily_ic.mean()),
            "n_oos_days": int(len(daily_ic)),
            **{
                k: summary[k]
                for k in (
                    "ann_return_net",
                    "ann_vol",
                    "sharpe_net",
                    "deflated_sharpe",
                    "hit_rate",
                    "max_drawdown",
                    "avg_daily_turnover",
                    "excess_ann_return_vs_spy",
                    "underperforms_spy",
                    "oos_start",
                    "oos_end",
                )
            },
        }
        if name != "incumbent_rank_all" and "incumbent_rank_all" in daily_net:
            results[name]["vs_incumbent"] = paired_stats(
                daily_net[name], daily_net["incumbent_rank_all"]
            )
        r = results[name]
        logger.info(
            "%-20s IC %+0.4f  ann_net %+6.2f%%  sharpe %+5.2f  dSR %.3f  maxDD %6.1f%%  TO %4.1f%%",
            name,
            r["mean_ic"],
            r["ann_return_net"] * 100,
            r["sharpe_net"],
            r["deflated_sharpe"],
            r["max_drawdown"] * 100,
            r["avg_daily_turnover"] * 100,
        )
        if "vs_incumbent" in r:
            s = r["vs_incumbent"]
            logger.info(
                "    vs incumbent: ddiff %+.2f bps/d (p=%.3f), dSharpe %+.3f, boot P(<=0)=%.3f",
                s["mean_daily_diff_bps"],
                s["p_value"],
                s["sharpe_diff_point"],
                s["bootstrap_p_sharpe_diff_le_0"],
            )

    out = {
        "n_trials": N_TRIALS_TOTAL,
        "coverage": event_coverage(data),
        "results": results,
    }
    (paths.ARTIFACTS / "experiment_wave6.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(daily_net).to_parquet(paths.ARTIFACTS / "experiment_wave6_daily.parquet")
    logger.info("wrote artifacts/experiment_wave6.json")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    np.random.seed(42)
    run_wave6(load_config(paths.CONFIGS / "baseline.yaml"))
