"""Performance metrics. Honest-reporting rules: deflated Sharpe, hit rate with a
confidence interval, and comparison vs. SPY buy-and-hold — always net of costs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


def sharpe(returns: pd.Series) -> float:
    sd = returns.std()
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(returns.mean() / sd * np.sqrt(TRADING_DAYS))


def deflated_sharpe(returns: pd.Series, n_trials: int = 1) -> float:
    """Probabilistic Sharpe ratio deflated for multiple testing
    (Bailey & López de Prado 2014). Returns P(true SR > 0) in [0, 1].
    """
    n = len(returns)
    if n < 20:
        return float("nan")
    sr_daily = returns.mean() / returns.std() if returns.std() > 0 else 0.0
    skew = float(stats.skew(returns, bias=False))
    kurt = float(stats.kurtosis(returns, fisher=False, bias=False))

    # Benchmark SR: expected max SR of n_trials random strategies (0 when n_trials == 1)
    if n_trials > 1:
        var_sr = 1.0 / (n - 1)
        emc = 0.5772156649
        sr0 = np.sqrt(var_sr) * (
            (1 - emc) * stats.norm.ppf(1 - 1 / n_trials)
            + emc * stats.norm.ppf(1 - 1 / (n_trials * np.e))
        )
    else:
        sr0 = 0.0

    denom = np.sqrt(1 - skew * sr_daily + (kurt - 1) / 4 * sr_daily**2)
    if denom <= 0 or np.isnan(denom):
        return float("nan")
    z = (sr_daily - sr0) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def hit_rate_ci(returns: pd.Series, alpha: float = 0.05) -> dict[str, float]:
    """Share of positive days with a Wilson score interval."""
    active = returns[returns != 0]
    n = len(active)
    if n == 0:
        return {"hit_rate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    wins = int((active > 0).sum())
    p = wins / n
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return {"hit_rate": p, "ci_low": center - half, "ci_high": center + half, "n": n}


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def summarize(
    net_returns: pd.Series,
    benchmark_returns: pd.Series,
    turnover: pd.Series,
    n_trials: int = 1,
) -> dict:
    equity = (1 + net_returns).cumprod()
    bench_equity = (1 + benchmark_returns.reindex(net_returns.index).fillna(0)).cumprod()
    years = len(net_returns) / TRADING_DAYS
    ann_ret = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else float("nan")
    bench_ann = float(bench_equity.iloc[-1] ** (1 / years) - 1) if years > 0 else float("nan")
    hit = hit_rate_ci(net_returns)
    return {
        "n_days": int(len(net_returns)),
        "ann_return_net": ann_ret,
        "ann_vol": float(net_returns.std() * np.sqrt(TRADING_DAYS)),
        "sharpe_net": sharpe(net_returns),
        "deflated_sharpe": deflated_sharpe(net_returns, n_trials=n_trials),
        "hit_rate": hit["hit_rate"],
        "hit_rate_ci95": [hit["ci_low"], hit["ci_high"]],
        "max_drawdown": max_drawdown(equity),
        "avg_daily_turnover": float(turnover.mean()),
        "benchmark_ann_return": bench_ann,
        "excess_ann_return_vs_spy": ann_ret - bench_ann,
        "underperforms_spy": bool(ann_ret < bench_ann),
    }
