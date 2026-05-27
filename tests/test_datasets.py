from recommender_systems.datasets import (
    load_goodbooks_10k,
    load_goodbooks_books,
    load_goodbooks_tags,
    load_movielens_100k,
)


def test_movielens_loads_cached_ratings_without_download(tmp_path):
    extracted = tmp_path / "ml-100k"
    extracted.mkdir(parents=True)
    (extracted / "u.data").write_text(
        "1\t10\t5\t881250949\n1\t20\t3\t881250949\n2\t10\t4\t881250949\n"
    )

    ratings = load_movielens_100k(data_home=tmp_path)

    assert list(ratings.columns) == ["user_id", "item_id", "rating", "timestamp"]
    assert len(ratings) == 3
    assert ratings.loc[0, "rating"] == 5


def _seed_goodbooks(root):
    d = root / "goodbooks-10k"
    d.mkdir(parents=True)
    (d / "ratings.csv").write_text("user_id,book_id,rating\n1,1,5\n1,2,3\n2,1,4\n")
    (d / "books.csv").write_text(
        "book_id,goodreads_book_id,title\n1,100,Book One\n2,200,Book Two\n"
    )
    (d / "book_tags.csv").write_text("goodreads_book_id,tag_id,count\n100,10,5\n200,20,3\n")
    (d / "tags.csv").write_text("tag_id,tag_name\n10,fantasy\n20,romance\n")


def test_goodbooks_ratings_and_books_load_offline(tmp_path):
    _seed_goodbooks(tmp_path)

    ratings = load_goodbooks_10k(data_home=tmp_path)
    assert list(ratings.columns) == ["user_id", "book_id", "rating"]
    assert len(ratings) == 3

    books = load_goodbooks_books(data_home=tmp_path)
    assert "title" in books.columns


def test_goodbooks_tags_join_to_book_id(tmp_path):
    _seed_goodbooks(tmp_path)

    tags = load_goodbooks_tags(data_home=tmp_path)

    assert list(tags.columns) == ["book_id", "tag_name", "count"]
    # book_id 1 -> goodreads_book_id 100 -> tag "fantasy"
    assert tags.loc[tags["book_id"] == 1, "tag_name"].iloc[0] == "fantasy"
