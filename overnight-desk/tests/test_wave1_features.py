"""Known-value tests for wave-1 feature families (fracdiff, spillover) and the
lambdarank grade construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.fracdiff import ffd_weights
from features.spillover import _eigenvector_centrality, _neighbor_weight_matrix


def test_ffd_weights_integer_orders_recover_classics():
    # d=1 -> plain first difference [1, -1, 0, ...]
    w1 = ffd_weights(1.0, 5)
    assert np.allclose(w1, [1, -1, 0, 0, 0])
    # d=0 -> identity [1, 0, ...]
    w0 = ffd_weights(0.0, 5)
    assert np.allclose(w0, [1, 0, 0, 0, 0])


def test_ffd_weights_fractional_decay():
    w = ffd_weights(0.4, 63)
    assert w[0] == 1.0
    assert w[1] == -0.4
    # weights alternate toward zero with slow decay: |w_k| strictly decreasing after k=1
    mags = np.abs(w[1:])
    assert (np.diff(mags) < 0).all()
    # memory preservation: far tail still nonzero (unlike d=1)
    assert abs(w[-1]) > 1e-6


def test_neighbor_weight_matrix_topk():
    corr = pd.DataFrame(
        [[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]],
        index=list("ABC"),
        columns=list("ABC"),
    )
    w = _neighbor_weight_matrix(corr, k=1)
    assert w.loc["A", "B"] == 1.0  # B is A's top neighbor
    assert w.loc["A", "A"] == 0.0  # self excluded
    assert w.loc["C", "B"] == 1.0  # B is C's top neighbor
    assert (w.sum(axis=1) == 1.0).all()


def test_eigenvector_centrality_hub_dominates():
    # star network: A correlated with everyone, B/C/D not with each other
    corr = pd.DataFrame(
        [
            [1.0, 0.8, 0.8, 0.8],
            [0.8, 1.0, 0.0, 0.0],
            [0.8, 0.0, 1.0, 0.0],
            [0.8, 0.0, 0.0, 1.0],
        ],
        index=list("ABCD"),
        columns=list("ABCD"),
    )
    cent = _eigenvector_centrality(corr)
    assert cent["A"] > cent["B"]
    assert np.isclose(cent.sum(), 1.0)


def test_lambdarank_grades_preserve_within_date_order():
    from core.config import Config
    from models.train import N_RANK_GRADES

    df = pd.DataFrame(
        {
            "date": pd.Timestamp("2024-01-02"),
            "label": np.linspace(-0.5, 0.5, 10),
        }
    )
    grades = (
        df.groupby("date")["label"]
        .rank(pct=True)
        .mul(N_RANK_GRADES)
        .clip(upper=N_RANK_GRADES - 1e-9)
        .astype(int)
    )
    assert grades.min() == 0 and grades.max() == N_RANK_GRADES - 1
    assert (grades.diff().dropna() >= 0).all()  # monotone in label
    assert Config().model.objective == "regression"  # default unchanged
