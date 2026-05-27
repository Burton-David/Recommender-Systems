import numpy as np
import pandas as pd
import pytest

from recommender_systems import build_user_item_matrix, split_interactions


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
    train, test = split_interactions(ratings, test_size=0.5, random_state=0)
    assert len(test) == 2
    assert len(train) + len(test) == len(ratings)


def test_split_is_reproducible(ratings):
    a_train, a_test = split_interactions(ratings, test_size=0.5, random_state=42)
    b_train, b_test = split_interactions(ratings, test_size=0.5, random_state=42)
    pd.testing.assert_frame_equal(a_train, b_train)
    pd.testing.assert_frame_equal(a_test, b_test)


def test_split_invalid_size_raises(ratings):
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="test_size"):
            split_interactions(ratings, test_size=bad)
