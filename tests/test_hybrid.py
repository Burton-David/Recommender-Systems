import pandas as pd
import pytest

from recommender_systems.base import Recommender
from recommender_systems.hybrid import HybridRecommender


class _Fixed(Recommender):
    """A recommender that always returns a fixed ranked list, for testing fusion."""

    def __init__(self, items):
        self._items = list(items)

    def fit(self, ratings):
        return self

    def recommend(self, user_id, n=10):
        return self._items[:n]


def test_hybrid_implements_interface():
    assert issubclass(HybridRecommender, Recommender)


def test_fusion_lifts_items_ranked_by_multiple_components():
    # "y" is the only item both components recommend, so fusion should rank it first.
    a = _Fixed(["x", "y"])
    b = _Fixed(["y", "z"])
    model = HybridRecommender([a, b]).fit(pd.DataFrame())
    assert model.recommend(1, n=1) == ["y"]


def test_weights_shift_the_ranking():
    a = _Fixed(["x"])
    b = _Fixed(["z"])
    # Weighting the second component far higher puts its pick on top.
    model = HybridRecommender([a, b], weights=[1.0, 10.0]).fit(pd.DataFrame())
    assert model.recommend(1, n=1) == ["z"]


def test_requires_at_least_one_recommender():
    with pytest.raises(ValueError, match="at least one"):
        HybridRecommender([])


def test_weights_must_match_recommenders():
    with pytest.raises(ValueError, match="weights must match"):
        HybridRecommender([_Fixed(["a"])], weights=[1.0, 2.0])
