import numpy as np
import pandas as pd

from recommender_systems.base import Recommender
from recommender_systems.bpr import BPR


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


def test_bpr_implements_interface():
    assert issubclass(BPR, Recommender)


def test_bpr_separates_clusters():
    model = BPR(n_factors=8, epochs=40, learning_rate=0.1, random_state=0)
    model.fit(_two_cluster_implicit_feedback())
    # User 0 is in cluster 0 (items 0-24); their top recommendations should be
    # dominated by cluster-0 items.
    recs = model.recommend(0, n=10)
    cluster_0_hits = sum(1 for r in recs if r < 25)
    assert cluster_0_hits >= 8, f"only {cluster_0_hits}/10 recommendations from the right cluster"


def test_bpr_excludes_seen():
    ratings = _two_cluster_implicit_feedback()
    model = BPR(n_factors=4, epochs=10, random_state=0).fit(ratings)
    user_id = 0
    seen = set(ratings[ratings["user_id"] == user_id]["item_id"])
    recs = model.recommend(user_id, n=20)
    assert not (set(recs) & seen)


def test_bpr_unknown_user_returns_empty():
    model = BPR(n_factors=4, epochs=5, random_state=0).fit(_two_cluster_implicit_feedback())
    assert model.recommend(9999, n=5) == []


def test_bpr_is_reproducible():
    ratings = _two_cluster_implicit_feedback()
    a = BPR(n_factors=4, epochs=10, random_state=42).fit(ratings).recommend(0, n=5)
    b = BPR(n_factors=4, epochs=10, random_state=42).fit(ratings).recommend(0, n=5)
    assert a == b
