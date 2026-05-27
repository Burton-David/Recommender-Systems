import pandas as pd

from recommender_systems.base import Recommender
from recommender_systems.neighborhood import ItemKNN, UserKNN


def sample_ratings():
    # users 1 and 2 overlap on a, b; user 2 also rated c; user 3 is isolated on d.
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 2, 3],
            "item_id": ["a", "b", "a", "b", "c", "d"],
            "rating": [5, 5, 5, 5, 5, 5],
        }
    )


def test_knn_implement_interface():
    assert issubclass(ItemKNN, Recommender)
    assert issubclass(UserKNN, Recommender)


def test_item_knn_recommends_cooccurring_item():
    # c co-occurs with a and b, which user 1 has rated -> c is the top unseen item.
    model = ItemKNN().fit(sample_ratings())
    assert model.recommend(1, n=1) == ["c"]


def test_user_knn_recommends_from_nearest_neighbor():
    # user 2 is user 1's nearest neighbor and rated c -> c is recommended.
    model = UserKNN().fit(sample_ratings())
    assert model.recommend(1, n=1) == ["c"]


def test_unknown_user_returns_empty():
    assert UserKNN().fit(sample_ratings()).recommend(999) == []
