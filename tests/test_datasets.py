from recommender_systems.datasets import load_movielens_100k


def test_loads_cached_ratings_without_download(tmp_path):
    extracted = tmp_path / "ml-100k"
    extracted.mkdir(parents=True)
    (extracted / "u.data").write_text(
        "1\t10\t5\t881250949\n1\t20\t3\t881250949\n2\t10\t4\t881250949\n"
    )

    ratings = load_movielens_100k(data_home=tmp_path)

    assert list(ratings.columns) == ["user_id", "item_id", "rating", "timestamp"]
    assert len(ratings) == 3
    assert ratings.loc[0, "rating"] == 5
