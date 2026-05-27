import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from recommender_systems import (
    build_sparse_user_item_matrix,
    build_user_item_matrix,
    densest_subset,
    holdout_per_user,
    split_ratings,
)


@pytest.fixture
def ratings():
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 3],
            "item_id": ["a", "b", "a", "c"],
            "rating": [5.0, 3.0, 4.0, 2.0],
        }
    )


def test_matrix_shape_and_values(ratings):
    matrix = build_user_item_matrix(ratings)
    assert matrix.shape == (3, 3)
    assert matrix.loc[1, "a"] == 5.0
    assert np.isnan(matrix.loc[2, "b"])


def test_matrix_fill_value(ratings):
    matrix = build_user_item_matrix(ratings, fill_value=0.0)
    assert matrix.loc[2, "b"] == 0.0


def test_matrix_averages_duplicate_pairs():
    ratings = pd.DataFrame({"user_id": [1, 1], "item_id": ["a", "a"], "rating": [2.0, 4.0]})
    matrix = build_user_item_matrix(ratings)
    assert matrix.loc[1, "a"] == 3.0


def test_matrix_missing_column_raises():
    with pytest.raises(ValueError, match="missing required columns"):
        build_user_item_matrix(pd.DataFrame({"user_id": [1], "item_id": ["a"]}))


def test_split_sizes_and_partition(ratings):
    train, test = split_ratings(ratings, test_size=0.5, random_state=0)
    assert len(test) == 2
    assert len(train) + len(test) == len(ratings)


def test_split_is_reproducible(ratings):
    a_train, a_test = split_ratings(ratings, test_size=0.5, random_state=42)
    b_train, b_test = split_ratings(ratings, test_size=0.5, random_state=42)
    pd.testing.assert_frame_equal(a_train, b_train)
    pd.testing.assert_frame_equal(a_test, b_test)


def test_split_invalid_size_raises(ratings):
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="test_size"):
            split_ratings(ratings, test_size=bad)


def test_densest_subset_keeps_top_users_and_items():
    rows = []
    # users 1 and 2 are active; user 3 rates once. items a, b are popular; c is rare.
    for _ in range(3):
        rows += [(1, "a"), (2, "b")]
    rows += [(1, "b"), (2, "a"), (3, "c")]
    ratings = pd.DataFrame(rows, columns=["user_id", "item_id"]).assign(rating=1)

    subset = densest_subset(ratings, n_users=2, n_items=2)

    assert set(subset["user_id"]) == {1, 2}
    assert set(subset["item_id"]) == {"a", "b"}


def test_holdout_per_user_keeps_every_user_in_train():
    rows = [(u, i) for u in range(5) for i in range(10)]  # 5 users x 10 items
    rows.append((99, 0))  # a single-interaction user
    ratings = pd.DataFrame(rows, columns=["user_id", "item_id"]).assign(rating=1)

    train, test = holdout_per_user(ratings, test_size=0.2, random_state=0)

    # every user appears in train (no cold users); the singleton stays out of test
    assert set(train["user_id"]) == set(ratings["user_id"])
    assert 99 not in set(test["user_id"])
    # each multi-item user has ~2 of 10 held out
    assert (test["user_id"] == 0).sum() == 2
    assert len(train) + len(test) == len(ratings)


def test_holdout_per_user_is_reproducible():
    ratings = pd.DataFrame(
        [(u, i) for u in range(4) for i in range(8)], columns=["user_id", "item_id"]
    ).assign(rating=1)
    a = holdout_per_user(ratings, random_state=7)
    b = holdout_per_user(ratings, random_state=7)
    pd.testing.assert_frame_equal(a[1], b[1])


def test_build_sparse_user_item_matrix_shape_values_and_maps():
    ratings = pd.DataFrame(
        {"user_id": [1, 1, 2], "item_id": ["a", "b", "a"], "rating": [5.0, 3.0, 4.0]}
    )
    matrix, users, items = build_sparse_user_item_matrix(ratings)

    assert sparse.issparse(matrix)
    assert matrix.shape == (len(users), len(items)) == (2, 2)
    assert list(users) == [1, 2]
    assert list(items) == ["a", "b"]
    assert matrix[users.get_loc(1), items.get_loc("b")] == 3.0
    assert matrix[users.get_loc(2), items.get_loc("a")] == 4.0
    assert matrix[users.get_loc(2), items.get_loc("b")] == 0.0


def test_build_sparse_averages_duplicate_pairs():
    ratings = pd.DataFrame({"user_id": [1, 1], "item_id": ["a", "a"], "rating": [2.0, 4.0]})
    matrix, _users, _items = build_sparse_user_item_matrix(ratings)
    assert matrix[0, 0] == 3.0
