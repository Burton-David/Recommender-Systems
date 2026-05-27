import pandas as pd

from recommender_systems.base import Recommender
from recommender_systems.baselines import MeanRating, MostPopular


def sample_ratings():
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3, 3],
            "item_id": ["a", "b", "a", "c", "a", "b", "c"],
            "rating": [5, 3, 4, 2, 5, 1, 4],
        }
    )


def test_baselines_implement_interface():
    assert issubclass(MostPopular, Recommender)
    assert issubclass(MeanRating, Recommender)


def test_most_popular_ranks_by_count_and_excludes_seen():
    model = MostPopular().fit(sample_ratings())
    # counts: a=3, b=2, c=2 -> ranking [a, b, c]; user 1 has seen {a, b}
    assert model.recommend(1, n=10) == ["c"]
    assert model.recommend(999, n=2) == ["a", "b"]


def test_mean_rating_ranks_by_mean_and_excludes_seen():
    model = MeanRating().fit(sample_ratings())
    # means: a=14/3, c=3.0, b=2.0 -> ranking [a, c, b]; user 2 has seen {a, c}
    assert model.recommend(2, n=10) == ["b"]
    assert model.recommend(999, n=2) == ["a", "c"]


def test_mean_rating_min_ratings_filters_items():
    model = MeanRating(min_ratings=3).fit(sample_ratings())
    # only item "a" has at least 3 ratings
    assert model.recommend(999, n=10) == ["a"]
