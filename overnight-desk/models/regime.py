"""Two-state statistical jump model for market regime detection.

Simplified Nystrup/Kolm-style jump model: k-means-like coordinate descent on daily
market features with an explicit jump penalty λ on state switches — the penalty is
what stops the HMM-style rapid state flickering that makes regime gates useless.

Point-in-time protocol: the model is refit every REFIT sessions on a trailing
FIT_WINDOW of features ending at the refit date (feature standardization params from
that window only). Between refits, new days are assigned online with frozen centroids
and the same switching penalty. The state at date t therefore uses only data <= t.

Deterministic: initialization is a median split on the volatility feature (no random
seeds), and coordinate descent is order-stable.

States: 0 = calm, 1 = stress (the centroid with higher standardized volatility).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FIT_WINDOW = 1008  # ~4 trading years
REFIT = 21
MAX_ITER = 20


def market_features(returns: pd.Series) -> pd.DataFrame:
    """Daily features from a market return series (index=date): 10d mean return,
    10d realized vol, 10d downside deviation."""
    feats = pd.DataFrame(index=returns.index)
    feats["mean10"] = returns.rolling(10, min_periods=5).mean()
    feats["vol10"] = returns.rolling(10, min_periods=5).std()
    downside = returns.clip(upper=0.0)
    feats["down10"] = downside.rolling(10, min_periods=5).std()
    return feats.dropna()


def _viterbi_assign(x: np.ndarray, centroids: np.ndarray, penalty: float) -> np.ndarray:
    """Optimal state path minimizing sum of squared distances + penalty per switch."""
    n, k = len(x), len(centroids)
    dist = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    cost = np.full((n, k), np.inf)
    back = np.zeros((n, k), dtype=int)
    cost[0] = dist[0]
    for t in range(1, n):
        for s in range(k):
            trans = cost[t - 1] + penalty * (np.arange(k) != s)
            j = int(np.argmin(trans))
            cost[t, s] = dist[t, s] + trans[j]
            back[t, s] = j
    path = np.zeros(n, dtype=int)
    path[-1] = int(np.argmin(cost[-1]))
    for t in range(n - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path


def fit_jump_model(
    feats: pd.DataFrame, penalty: float
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """Fit 2-state jump model. Returns (centroids, states, mean, std) — mean/std are
    the standardization params to reuse for online assignment."""
    mean, std = feats.mean(), feats.std().replace(0, 1e-12)
    x = ((feats - mean) / std).to_numpy()

    # Deterministic init: median split on standardized vol10
    vol_ix = list(feats.columns).index("vol10")
    states = (x[:, vol_ix] > np.median(x[:, vol_ix])).astype(int)

    for _ in range(MAX_ITER):
        centroids = np.vstack(
            [x[states == s].mean(axis=0) if (states == s).any() else x.mean(axis=0) for s in (0, 1)]
        )
        new_states = _viterbi_assign(x, centroids, penalty)
        if (new_states == states).all():
            break
        states = new_states

    # Convention: state 1 = stress = higher standardized vol centroid
    if centroids[0, vol_ix] > centroids[1, vol_ix]:
        centroids = centroids[::-1]
        states = 1 - states
    return centroids, states, mean, std


def regime_series(returns: pd.Series, penalty: float = 50.0) -> pd.Series:
    """Point-in-time regime state (0 calm / 1 stress) for every date with enough
    history. Refit every REFIT sessions on a trailing FIT_WINDOW; assign the block
    after each refit online with frozen centroids, chaining from the fitted path."""
    feats = market_features(returns)
    out = pd.Series(np.nan, index=feats.index)

    for start in range(FIT_WINDOW, len(feats), REFIT):
        window = feats.iloc[start - FIT_WINDOW : start]
        centroids, states, mean, std = fit_jump_model(window, penalty)
        block = feats.iloc[start : min(start + REFIT, len(feats))]
        x = ((block - mean) / std.replace(0, 1e-12)).to_numpy()
        prev = int(states[-1])
        assigned = []
        for row in x:
            dist = ((row - centroids) ** 2).sum(axis=1)
            dist += penalty * (np.arange(2) != prev)
            prev = int(np.argmin(dist))
            assigned.append(prev)
        out.iloc[start : start + len(assigned)] = assigned

    return out.dropna().astype(int)
