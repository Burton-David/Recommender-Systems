import pandas as pd

from recommender_systems.books import build_tag_recommender, tag_text_per_book


def _sample_tags():
    # Books 1 and 2 share the "fantasy" tag; 3 and 4 share "romance".
    return pd.DataFrame(
        {
            "book_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "tag_name": [
                "fantasy",
                "magic",
                "fantasy",
                "epic",
                "romance",
                "regency",
                "romance",
                "contemporary",
            ],
            "count": [10, 5, 8, 4, 6, 3, 7, 2],
        }
    )


def test_tag_text_per_book_joins_tags():
    text = tag_text_per_book(_sample_tags())
    assert {1, 2, 3, 4} == set(text)
    assert "fantasy" in text[1]
    assert "magic" in text[1]
    assert "romance" in text[3]


def test_build_tag_recommender_uses_tag_similarity():
    tags = _sample_tags()
    # User 1 rated the first fantasy book; the recommender should rank the other
    # fantasy book ahead of the romance ones.
    ratings = pd.DataFrame({"user_id": [1], "item_id": [1], "rating": [5]})
    model = build_tag_recommender(tags).fit(ratings)
    recs = model.recommend(1, n=3)
    assert recs[0] == 2
    assert 3 not in recs[:1]
    assert 4 not in recs[:1]


def test_vectorizer_kwargs_are_forwarded():
    # min_df=2 drops tags that appear in only one book — leaves only "fantasy"
    # and "romance" in the vocabulary. The recommender still runs end-to-end.
    tags = _sample_tags()
    ratings = pd.DataFrame({"user_id": [1], "item_id": [1], "rating": [5]})
    model = build_tag_recommender(tags, min_df=2).fit(ratings)
    # User 1 liked book 1 (fantasy); the only other fantasy book is 2.
    assert model.recommend(1, n=1) == [2]
