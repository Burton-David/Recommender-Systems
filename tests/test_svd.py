import pandas as pd
from scipy import sparse

from recommender_systems.base import Recommender
from recommender_systems.svd import SVD


def sample_ratings():
    # Two latent groups: {a, b} liked together, {c} separate. User 4 rated only a.
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 4],
            "item_id": ["a", "b", "a", "b", "c", "a"],
            "rating": [5, 5, 5, 5, 5, 5],
        }
    )


def test_svd_implements_interface():
    assert issubclass(SVD, Recommender)


def test_svd_recovers_latent_preference():
    # User 4 loads on the {a, b} factor, so b should be recommended over c.
    model = SVD(random_state=0).fit(sample_ratings())
    assert model.recommend(4, n=1) == ["b"]


def test_svd_excludes_seen_and_handles_unknown_user():
    model = SVD(random_state=0).fit(sample_ratings())
    assert "a" not in model.recommend(4)
    assert model.recommend(999) == []


def test_svd_stores_sparse_matrix():
    # Regression: pins the sparse internal representation so a future change
    # can't silently revert to dense and reintroduce the goodbooks-10k memory
    # blowup the Phase 3 rewrite fixed.
    model = SVD(random_state=0).fit(sample_ratings())
    assert sparse.issparse(model._matrix)
