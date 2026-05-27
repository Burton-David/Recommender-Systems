"""Regression tests for the Rust-backed BPR training path.

Skipped when the compiled ``_kernels`` extension isn't importable — the same
condition under which ``BPR.fit`` falls back to the pure-Python loop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("recommender_systems._kernels")

from recommender_systems import _kernels
from recommender_systems.bpr import BPR
from recommender_systems.data import split_ratings
from recommender_systems.datasets import load_movielens_100k
from recommender_systems.metrics import precision_at_k, recall_at_k


def test_bpr_train_kernel_is_callable() -> None:
    """Smoke test the PyO3 binding signature on tiny inputs."""
    n_users, n_items, n_factors = 4, 5, 3
    user_factors = np.zeros((n_users, n_factors), dtype=np.float64)
    item_factors = np.zeros((n_items, n_factors), dtype=np.float64)
    positives = np.array([[0, 0], [0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64)
    observed = np.zeros((n_users, n_items), dtype=bool)
    for u, i in positives:
        observed[u, i] = True

    _kernels.bpr_train(
        user_factors,
        item_factors,
        positives,
        observed.reshape(-1),
        n_items,
        epochs=1,
        learning_rate=0.05,
        reg=0.01,
        seed=0,
    )
    # Factors should have moved from zero.
    assert np.any(user_factors != 0.0)
    assert np.any(item_factors != 0.0)


def test_kernel_backed_bpr_matches_python_on_quality() -> None:
    """The Rust kernel produces a different exact byte stream from the Python
    loop (different RNG ordering), but recommendation quality on MovieLens 100k
    should be within tolerance of the Python baseline.

    Tolerance is loose because BPR with 5 epochs is itself noisy across seeds.
    The real-world claim is "Rust kernel doesn't regress recommendation
    quality"; this test enforces that as a soft floor.
    """
    ratings = load_movielens_100k()
    train, test = split_ratings(ratings, test_size=0.2, random_state=0)

    model = BPR(n_factors=16, epochs=5, random_state=0).fit(train)

    truth = test.groupby("user_id")["item_id"].agg(set)
    users = sorted(truth.index)
    predicted = [model.recommend(u, n=10) for u in users]
    actual = [truth[u] for u in users]

    # The Python baseline lands around precision@10 ~ 0.10-0.14 with these
    # hyperparameters; we require the kernel path to clear a conservative floor.
    p10 = precision_at_k(predicted, actual, k=10)
    r10 = recall_at_k(predicted, actual, k=10)
    assert p10 > 0.05, f"precision@10 = {p10:.4f} below tolerance floor"
    assert r10 > 0.02, f"recall@10 = {r10:.4f} below tolerance floor"


def test_bpr_reproducible_for_a_given_seed() -> None:
    """Same seed → same model state (same factor arrays after training)."""
    rng = np.random.default_rng(0)
    n_pairs = 200
    user_ids = rng.integers(0, 20, size=n_pairs)
    item_ids = rng.integers(0, 30, size=n_pairs)
    ratings = pd.DataFrame(
        {
            "user_id": user_ids,
            "item_id": item_ids,
            "rating": np.ones(n_pairs),
        }
    ).drop_duplicates(subset=["user_id", "item_id"])

    a = BPR(n_factors=8, epochs=3, random_state=42).fit(ratings)
    b = BPR(n_factors=8, epochs=3, random_state=42).fit(ratings)
    np.testing.assert_allclose(a._user_factors, b._user_factors)
    np.testing.assert_allclose(a._item_factors, b._item_factors)
