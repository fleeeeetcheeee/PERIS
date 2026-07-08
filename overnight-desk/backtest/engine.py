"""Vectorized daily backtester with the transaction-cost model always on.

Timing convention: signals are computed from data through close t; target weights
are set at close t and earn the close(t) -> close(t+1) return. Costs are charged
per side on traded weight (7.5 bps stocks / 3 bps ETFs by default). Realized
slippage vs. this model is reconciled monthly from the ledger.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.config import Config
from portfolio import construct
from portfolio.gates import (
    GateDecision,
    drawdown_gate,
    jump_regime_gate,
    vix_term_structure_gate,
)

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    net_returns: pd.Series
    gross_returns: pd.Series  # internal only — never a headline number
    costs: pd.Series
    turnover: pd.Series
    weights: pd.DataFrame
    equity: pd.Series


def _cost_bps(asset_types: dict[str, str], cfg: Config) -> pd.Series:
    return pd.Series(
        {
            t: (cfg.costs.etf_bps_per_side if a == "etf" else cfg.costs.stock_bps_per_side)
            for t, a in asset_types.items()
        }
    )


def _meta_sizing(
    day: pd.DataFrame | None, port_cfg
) -> tuple[pd.Series | None, bool]:
    """Per-mode conviction multipliers for one rebalance date.

    Returns (size_mult, renormalize). None means neutral — either meta is off or
    the meta walk-forward has no probabilities yet for this date (warm-up).
    """
    mode = port_cfg.meta_mode
    if mode == "off" or day is None:
        return None, False
    if mode == "tilt":
        return day["meta_prob"], True
    if mode == "gate":
        return (day["meta_prob"] >= port_cfg.meta_gate_threshold).astype(float), False
    if mode == "sized":
        from models.meta import size_multiplier

        base = float(day["meta_base_rate"].max())  # per-date constant (NaN-safe)
        return size_multiplier(day["meta_prob_cal"], base), False
    raise ValueError(f"unknown meta_mode: {mode!r}")


def run_backtest(
    predictions: pd.DataFrame,  # columns: date, ticker, score
    prices: pd.DataFrame,  # long panel: date, ticker, close
    asset_types: dict[str, str],
    cfg: Config,
    macro_vix: pd.DataFrame | None = None,  # columns: date, VIX, VIX3M (already lagged)
    regime: pd.Series | None = None,  # date -> 0 calm / 1 stress (PIT jump model)
    meta: pd.DataFrame | None = None,  # models.meta probabilities (walk-forward OOS)
) -> BacktestResult:
    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    rets = close.pct_change()
    vol63 = rets.rolling(63, min_periods=40).std() * np.sqrt(252)
    cost_bps = _cost_bps(asset_types, cfg)

    if regime is None and cfg.gates.jump_model and cfg.benchmark in close.columns:
        from models.regime import regime_series

        regime = regime_series(
            close[cfg.benchmark].pct_change().dropna(), penalty=cfg.gates.jump_penalty
        )

    pred_dates = sorted(predictions["date"].unique())
    if cfg.portfolio.score_smoothing_halflife > 0:
        # EMA per ticker over prediction dates — uses only past scores, so no lookahead.
        wide_scores = predictions.pivot(index="date", columns="ticker", values="score").sort_index()
        wide_scores = wide_scores.ewm(
            halflife=cfg.portfolio.score_smoothing_halflife, min_periods=1
        ).mean()
        scores_by_date = {d: wide_scores.loc[d].dropna() for d in pred_dates}
    else:
        scores_by_date = {d: g.set_index("ticker")["score"] for d, g in predictions.groupby("date")}
    vix_by_date: dict = {}
    if macro_vix is not None and not macro_vix.empty:
        vix_by_date = {r["date"]: (r.get("VIX"), r.get("VIX3M")) for _, r in macro_vix.iterrows()}

    meta_by_date: dict = {}
    if cfg.portfolio.meta_mode != "off" and meta is not None and not meta.empty:
        meta_by_date = {d: g.set_index("ticker") for d, g in meta.groupby("date")}

    dates_idx = close.index
    prev_w = pd.Series(dtype=float)
    equity = 1.0
    # Trailing (252-session) high-water mark: an all-time HWM makes the drawdown
    # breaker a one-way trap — once the book goes flat below the halt level, equity
    # can never recover and the strategy stays dead forever. A rolling HWM releases
    # the breaker after a recovery window.
    equity_history: list[float] = [1.0]
    rows = []
    weight_rows = {}

    sessions_since_rebalance = cfg.portfolio.rebalance_every  # rebalance on day one
    for t in pred_dates:
        if t not in dates_idx:
            continue
        loc = dates_idx.get_loc(t)
        if loc + 1 >= len(dates_idx):
            break  # no next-day return yet
        t1 = dates_idx[loc + 1]

        sessions_since_rebalance += 1
        if sessions_since_rebalance >= cfg.portfolio.rebalance_every:
            scores = scores_by_date[t].dropna()
            vols = vol63.loc[t]
            eligible = pd.Series(True, index=scores.index)
            eligible &= vols.reindex(scores.index).notna()
            eligible &= close.loc[t].reindex(scores.index).notna()

            decision = GateDecision()
            vix, vix3m = vix_by_date.get(t, (None, None))
            decision = vix_term_structure_gate(decision, vix, vix3m, cfg.gates)
            if cfg.gates.jump_model:
                state = int(regime.loc[t]) if regime is not None and t in regime.index else None
                decision = jump_regime_gate(decision, state, cfg.gates)
            hwm = max(equity_history[-252:])
            decision = drawdown_gate(decision, equity, hwm, cfg.gates)
            eligible &= ~eligible.index.isin(decision.excluded)

            size_mult, size_renorm = _meta_sizing(meta_by_date.get(t), cfg.portfolio)

            trailing = rets.iloc[max(0, loc - cfg.portfolio.vol_lookback_days) : loc + 1]
            target = construct.build_targets(
                scores=scores,
                vol=vols.reindex(scores.index),
                eligible=eligible,
                trailing_returns=trailing,
                previous=prev_w,
                cfg=cfg.portfolio,
                exposure_scale=decision.exposure_scale,
                allow_new_entries=decision.allow_new_entries,
                size_mult=size_mult,
                size_renormalize=size_renorm,
            )
            sessions_since_rebalance = 0
        else:
            target = prev_w  # drift between rebalances; no trades emitted

        all_names = target.index.union(prev_w.index)
        delta = target.reindex(all_names, fill_value=0.0) - prev_w.reindex(
            all_names, fill_value=0.0
        )
        turnover = float(delta.abs().sum())
        cost = float(
            (
                delta.abs() * cost_bps.reindex(all_names, fill_value=cfg.costs.stock_bps_per_side)
            ).sum()
            / 1e4
        )

        r_next = rets.loc[t1].reindex(target.index).fillna(0.0)
        gross = float((target * r_next).sum())
        net = gross - cost

        equity *= 1 + net
        equity_history.append(equity)
        rows.append({"date": t1, "gross": gross, "net": net, "cost": cost, "turnover": turnover})
        weight_rows[t] = target

        # drift weights to next close
        drifted = target * (1 + r_next)
        total = drifted.sum() + (1 - target.sum())  # cash earns 0
        prev_w = drifted / total if total > 0 else pd.Series(dtype=float)

    if not rows:
        raise RuntimeError("backtest produced no returns — check prediction/price alignment")
    df = pd.DataFrame(rows).set_index("date")
    equity_curve = (1 + df["net"]).cumprod()
    return BacktestResult(
        net_returns=df["net"],
        gross_returns=df["gross"],
        costs=df["cost"],
        turnover=df["turnover"],
        weights=pd.DataFrame(weight_rows).T.fillna(0.0),
        equity=equity_curve,
    )
