import pandas as pd
import pytest

from recommender_systems.content import ContentBased
from recommender_systems.features import text_features

ITEM_TEXT = {
    "a": "space opera science fiction",
    "b": "science fiction robots",
    "c": "regency romance",
}


def test_tfidf_shape_and_index():
    feats = text_features(ITEM_TEXT, method="tfidf")
    assert list(feats.index) == ["a", "b", "c"]
    assert "science" in feats.columns
    # "science fiction" items share weight on those terms; the romance item does not.
    assert feats.loc["a", "science"] > 0
    assert feats.loc["c", "science"] == 0


def test_binary_is_zero_or_one():
    feats = text_features(ITEM_TEXT, method="binary")
    assert set(feats.to_numpy().ravel()) <= {0, 1}


def test_count_matches_term_frequency():
    feats = text_features({"x": "alpha alpha beta"}, method="count")
    assert feats.loc["x", "alpha"] == 2
    assert feats.loc["x", "beta"] == 1


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown method"):
        text_features(ITEM_TEXT, method="word2vec")


def test_features_feed_content_based():
    feats = text_features(ITEM_TEXT, method="tfidf")
    ratings = pd.DataFrame({"user_id": [1], "item_id": ["a"], "rating": [5.0]})
    model = ContentBased(item_features=feats).fit(ratings)
    # Having liked the sci-fi item "a", the other sci-fi item "b" should rank above romance "c".
    recs = model.recommend(1, n=2)
    assert recs[0] == "b"
