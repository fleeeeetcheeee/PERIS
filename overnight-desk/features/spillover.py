"""Momentum-spillover features from the trailing correlation network.

The GNN stock-forecasting literature's core finding — a stock's correlated peers'
momentum predicts its own return — imported as three cheap features instead of a
parameterized graph model (our 71-name panel cannot feed a GNN):

- spill_neighbor_mom : average 126x5 momentum of the k most-correlated peers
- spill_peer_gap     : own 5d return minus neighbor-average 5d return (peer reversal)
- spill_centrality   : eigenvector centrality in the correlation network

Point-in-time contract: the correlation network is re-estimated every REFIT sessions
on a trailing CORR_WINDOW of market-demeaned daily returns ending at the refit date,
then held fixed until the next refit — a value at date t only ever uses data <= t.

Lookback: 126 (correlation window) + 131 (momentum) + 21 (refit staleness).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.common import by_ticker, daily_returns, validate_panel

LOOKBACK = 126 + 131 + 21

CORR_WINDOW = 126
REFIT = 21
K_NEIGHBORS = 5


def _neighbor_weight_matrix(corr: pd.DataFrame, k: int) -> pd.DataFrame:
    """Row-stochastic matrix W: W[i, j] = 1/k if j is one of i's top-k correlated
    peers (excluding self), else 0. neighbor_mean = feature_row @ W.T."""
    values = corr.to_numpy().copy()
    np.fill_diagonal(values, -np.inf)
    c = pd.DataFrame(values, index=corr.index, columns=corr.columns)
    w = pd.DataFrame(0.0, index=c.index, columns=c.columns)
    for ticker in c.index:
        neighbors = c.loc[ticker].nlargest(k).index
        w.loc[ticker, neighbors] = 1.0 / k
    return w


def _eigenvector_centrality(corr: pd.DataFrame) -> pd.Series:
    """Leading eigenvector of the non-negative correlation adjacency (diag zeroed)."""
    adj = corr.clip(lower=0.0).to_numpy().copy()
    np.fill_diagonal(adj, 0.0)
    eigvals, eigvecs = np.linalg.eigh(adj)
    lead = np.abs(eigvecs[:, -1])
    total = lead.sum()
    return pd.Series(lead / total if total > 0 else lead, index=corr.index)


def compute(panel: pd.DataFrame, macro: pd.DataFrame | None = None) -> pd.DataFrame:
    panel = validate_panel(panel)
    out = panel[["date", "ticker"]].copy()

    ret = daily_returns(panel)
    wide_ret = (
        pd.DataFrame({"date": panel["date"], "ticker": panel["ticker"], "ret": ret})
        .pivot(index="date", columns="ticker", values="ret")
        .sort_index()
    )
    demeaned = wide_ret.sub(wide_ret.mean(axis=1), axis=0)

    close_grp = by_ticker(panel)["close"]
    mom = close_grp.pct_change(126).shift(5)
    r5 = close_grp.pct_change(5)
    wide_mom = (
        pd.DataFrame({"date": panel["date"], "ticker": panel["ticker"], "v": mom})
        .pivot(index="date", columns="ticker", values="v")
        .sort_index()
    )
    wide_r5 = (
        pd.DataFrame({"date": panel["date"], "ticker": panel["ticker"], "v": r5})
        .pivot(index="date", columns="ticker", values="v")
        .sort_index()
    )

    dates = wide_ret.index
    nbr_mom = pd.DataFrame(np.nan, index=dates, columns=wide_ret.columns)
    nbr_r5 = pd.DataFrame(np.nan, index=dates, columns=wide_ret.columns)
    centrality = pd.DataFrame(np.nan, index=dates, columns=wide_ret.columns)

    # Refit the network on windows ENDING at each refit date; apply to the block
    # [refit date, next refit date) — strictly trailing.
    for start in range(CORR_WINDOW, len(dates), REFIT):
        window = demeaned.iloc[start - CORR_WINDOW : start]
        valid = window.columns[window.notna().sum() >= CORR_WINDOW // 2]
        if len(valid) < K_NEIGHBORS + 1:
            continue
        corr = window[valid].corr()
        w = _neighbor_weight_matrix(corr, K_NEIGHBORS)
        cent = _eigenvector_centrality(corr)

        block = slice(start, min(start + REFIT, len(dates)))
        nbr_mom.iloc[block, nbr_mom.columns.get_indexer(valid)] = (
            wide_mom.iloc[block][valid].fillna(0.0) @ w.T
        ).to_numpy()
        nbr_r5.iloc[block, nbr_r5.columns.get_indexer(valid)] = (
            wide_r5.iloc[block][valid].fillna(0.0) @ w.T
        ).to_numpy()
        centrality.iloc[block, centrality.columns.get_indexer(valid)] = cent.to_numpy()

    def melt(wide: pd.DataFrame, name: str) -> pd.Series:
        long = wide.stack().rename(name)
        keyed = out.set_index(["date", "ticker"]).index
        return long.reindex(keyed).to_numpy()

    out["spill_neighbor_mom"] = melt(nbr_mom, "nm")
    out["spill_peer_gap"] = wide_r5.stack().reindex(
        out.set_index(["date", "ticker"]).index
    ).to_numpy() - melt(nbr_r5, "nr")
    out["spill_centrality"] = melt(centrality, "ce")
    return out
