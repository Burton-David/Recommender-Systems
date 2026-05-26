import pandas as pd
import pytest

from recommender_systems import Recommender


def test_cannot_instantiate_abstract_base():
    with pytest.raises(TypeError):
        Recommender()


class _MostPopular(Recommender):
    """Minimal reference implementation used to exercise the interface."""

    def fit(self, ratings):
        self._ranking = (
            ratings.groupby("item_id").size().sort_values(ascending=False).index.tolist()
        )
        return self

    def recommend(self, user_id, n=10):
        return self._ranking[:n]


def test_reference_implementation_obeys_contract():
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3],
            "item_id": ["a", "b", "a", "c", "a"],
            "rating": [5, 4, 3, 5, 4],
        }
    )

    model = _MostPopular().fit(ratings)

    assert model.recommend(user_id=1, n=2) == ["a", "b"]
    assert model.recommend(user_id=1, n=10)[0] == "a"
