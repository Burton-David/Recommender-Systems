import pytest

pytest.importorskip("torch")  # full file is gated on torch being installed

import numpy as np
import pandas as pd

from recommender_systems.base import Recommender
from recommender_systems.neural import TwoTowerCF


def _two_cluster_implicit_feedback():
    """50 users in two clusters, each interacting with 6 items from their cluster."""
    rng = np.random.default_rng(0)
    rows = []
    for user in range(50):
        cluster = 0 if user < 25 else 1
        items = list(range(0, 25)) if cluster == 0 else list(range(25, 50))
        for item in rng.choice(items, size=6, replace=False):
            rows.append((user, int(item)))
    return pd.DataFrame(rows, columns=["user_id", "item_id"]).assign(rating=1)


def test_two_tower_implements_interface():
    assert issubclass(TwoTowerCF, Recommender)


def test_two_tower_separates_clusters():
    model = TwoTowerCF(n_factors=8, epochs=40, learning_rate=0.05, batch_size=64, random_state=0)
    model.fit(_two_cluster_implicit_feedback())
    recs = model.recommend(0, n=10)
    cluster_0_hits = sum(1 for r in recs if r < 25)
    assert cluster_0_hits >= 8, f"only {cluster_0_hits}/10 in the right cluster"


def test_two_tower_excludes_seen():
    ratings = _two_cluster_implicit_feedback()
    seen = set(ratings.loc[ratings["user_id"] == 0, "item_id"])
    model = TwoTowerCF(n_factors=4, epochs=5, random_state=0).fit(ratings)
    recs = model.recommend(0, n=20)
    assert not (set(recs) & seen)


def test_two_tower_unknown_user_returns_empty():
    model = TwoTowerCF(n_factors=4, epochs=5, random_state=0).fit(_two_cluster_implicit_feedback())
    assert model.recommend(9999, n=5) == []


def test_two_tower_is_reproducible():
    ratings = _two_cluster_implicit_feedback()
    a = TwoTowerCF(n_factors=4, epochs=5, random_state=42).fit(ratings).recommend(0, n=5)
    b = TwoTowerCF(n_factors=4, epochs=5, random_state=42).fit(ratings).recommend(0, n=5)
    assert a == b


def test_two_tower_terminates_when_a_user_has_rated_every_item():
    rows = [(0, item) for item in ("a", "b", "c")]
    rows += [(1, "a"), (2, "b")]
    ratings = pd.DataFrame(rows, columns=["user_id", "item_id"]).assign(rating=1)
    # Must complete without hanging on the per-batch resample loop.
    model = TwoTowerCF(n_factors=4, epochs=3, random_state=0).fit(ratings)
    assert model.recommend(0, n=5) == []
    assert model.recommend(1, n=2)
