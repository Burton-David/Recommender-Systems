import pandas as pd

from recommender_systems.base import Recommender
from recommender_systems.content import ContentBased


def sample_ratings():
    # user 1 rated action films a and b; user 2 rated drama c; item d is action.
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "item_id": ["a", "b", "c"],
            "rating": [5, 5, 5],
        }
    )


def sample_item_features():
    return pd.DataFrame(
        {"action": [1.0, 1.0, 0.0, 1.0], "drama": [0.0, 0.0, 1.0, 0.0]},
        index=["a", "b", "c", "d"],
    )


def test_content_based_implements_interface():
    assert issubclass(ContentBased, Recommender)


def test_constructor_takes_side_information():
    # The convention test: side information arrives via the constructor; fit takes
    # only the ratings frame, exactly like every other recommender in the library.
    model = ContentBased(item_features=sample_item_features())
    model.fit(sample_ratings())  # signature parity with all other recommenders
    assert model.recommend(1, n=1) == ["d"]


def test_recommends_by_feature_similarity():
    # User 1's profile is pure action; the only unseen action item is d.
    model = ContentBased(sample_item_features()).fit(sample_ratings())
    assert model.recommend(1, n=2) == ["d", "c"]


def test_excludes_seen_items():
    model = ContentBased(sample_item_features()).fit(sample_ratings())
    recs = model.recommend(1, n=10)
    assert "a" not in recs
    assert "b" not in recs


def test_unknown_user_returns_empty():
    model = ContentBased(sample_item_features()).fit(sample_ratings())
    assert model.recommend(999) == []


def test_handles_items_missing_from_features():
    # Items present in ratings but absent from item_features are zero-filled,
    # so they're still recommendable but score as having no signal.
    ratings = sample_ratings()
    partial_features = sample_item_features().drop(index=["c"])
    model = ContentBased(partial_features).fit(ratings)
    # User 1 still prefers d (the action item that's in the feature table).
    assert model.recommend(1, n=1) == ["d"]
