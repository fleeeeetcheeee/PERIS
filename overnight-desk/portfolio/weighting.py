"""Alternative weighting schemes for the selected top-K names.

- HRP (López de Prado 2016): hierarchical clustering on correlation distance,
  quasi-diagonalization, recursive bisection with inverse-variance cluster splits.
  No matrix inversion — robust to noisy covariance.
- RMT min-var: clip correlation eigenvalues inside the Marchenko-Pastur noise bulk
  to their mean (Bouchaud-Potters clipping), rebuild covariance, long-only
  minimum-variance weights.

Both consume a trailing returns window of the SELECTED names only (K ~ 12), and both
are deterministic. Long-only: any negative min-var weight is clipped to zero and the
rest renormalized.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


def _corr_cov(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = returns.dropna(axis=1, how="all").fillna(0.0)
    values = returns.corr().fillna(0.0).to_numpy().copy()
    np.fill_diagonal(values, 1.0)
    corr = pd.DataFrame(values, index=returns.columns, columns=returns.columns)
    vol = returns.std().replace(0, np.nan).fillna(returns.std().mean() or 1e-6)
    cov = corr * np.outer(vol, vol)
    return corr, pd.DataFrame(cov, index=corr.index, columns=corr.columns)


# --------------------------------------------------------------------- HRP


def _quasi_diag_order(corr: pd.DataFrame) -> list[str]:
    dist = np.sqrt(0.5 * (1 - corr.clip(-1, 1)))
    condensed = squareform(dist.to_numpy(), checks=False)
    link = linkage(condensed, method="single")
    return [corr.index[i] for i in leaves_list(link)]


def _cluster_var(cov: pd.DataFrame, names: list[str]) -> float:
    sub = cov.loc[names, names]
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub.to_numpy() @ ivp)


def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    """Hierarchical risk parity weights from a trailing returns window (cols=tickers)."""
    if returns.shape[1] == 1:
        return pd.Series(1.0, index=returns.columns)
    corr, cov = _corr_cov(returns)
    order = _quasi_diag_order(corr)
    weights = pd.Series(1.0, index=order)
    clusters: list[list[str]] = [order]
    while clusters:
        clusters = [
            half
            for cluster in clusters
            for half in (cluster[: len(cluster) // 2], cluster[len(cluster) // 2 :])
            if len(half) > 0 and len(cluster) > 1
        ]
        for i in range(0, len(clusters), 2):
            if i + 1 >= len(clusters):
                continue
            left, right = clusters[i], clusters[i + 1]
            var_left = _cluster_var(cov, left)
            var_right = _cluster_var(cov, right)
            alpha = 1 - var_left / (var_left + var_right)
            weights[left] *= alpha
            weights[right] *= 1 - alpha
    return weights / weights.sum()


# --------------------------------------------------------------------- RMT


def mp_clip_corr(corr: pd.DataFrame, n_obs: int) -> pd.DataFrame:
    """Clip eigenvalues below the Marchenko-Pastur upper edge to their average
    (preserving trace), keeping only signal eigenvalues intact."""
    n = corr.shape[0]
    q = n_obs / n
    lambda_max = (1 + np.sqrt(1.0 / q)) ** 2 if q > 0 else np.inf
    eigvals, eigvecs = np.linalg.eigh(corr.to_numpy())
    noise = eigvals < lambda_max
    if noise.any():
        eigvals = eigvals.copy()
        eigvals[noise] = eigvals[noise].mean()
    cleaned = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # renormalize to a correlation matrix
    d = np.sqrt(np.diag(cleaned))
    cleaned = cleaned / np.outer(d, d)
    np.fill_diagonal(cleaned, 1.0)
    return pd.DataFrame(cleaned, index=corr.index, columns=corr.columns)


def rmt_minvar_weights(returns: pd.DataFrame) -> pd.Series:
    """Long-only minimum-variance weights on the MP-cleaned covariance."""
    if returns.shape[1] == 1:
        return pd.Series(1.0, index=returns.columns)
    corr, _ = _corr_cov(returns)
    cleaned = mp_clip_corr(corr, n_obs=len(returns))
    vol = returns.fillna(0.0).std().replace(0, 1e-6)
    cov = cleaned.to_numpy() * np.outer(vol, vol)
    try:
        inv = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        return pd.Series(1.0 / returns.shape[1], index=returns.columns)
    ones = np.ones(cov.shape[0])
    w = inv @ ones
    w = np.clip(w, 0.0, None)  # long-only, cash account
    if w.sum() <= 0:
        w = ones
    return pd.Series(w / w.sum(), index=corr.index)
