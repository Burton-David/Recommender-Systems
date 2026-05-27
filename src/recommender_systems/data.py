"""Utilities for preparing interaction data for recommenders."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["build_user_item_matrix", "train_test_split"]


def build_user_item_matrix(
    ratings: pd.DataFrame,
    *,
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: str = "rating",
    fill_value: float | None = None,
) -> pd.DataFrame:
    """Pivot a long ratings frame into a user-by-item matrix.

    Parameters
    ----------
    ratings
        Long-format interactions with one row per (user, item) rating.
    user_col, item_col, rating_col
        Column names to read from ``ratings``.
    fill_value
        Value for user-item pairs with no rating. ``None`` (default) leaves them as NaN.

    Returns
    -------
    pandas.DataFrame
        Rows indexed by user, columns by item, values the rating. Duplicate
        (user, item) pairs are averaged.
    """
    missing = {user_col, item_col, rating_col} - set(ratings.columns)
    if missing:
        raise ValueError(f"ratings is missing required columns: {sorted(missing)}")
    matrix = ratings.pivot_table(
        index=user_col, columns=item_col, values=rating_col, aggfunc="mean"
    )
    if fill_value is not None:
        matrix = matrix.fillna(fill_value)
    return matrix


def train_test_split(
    ratings: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split interactions into train and test sets by random row sampling.

    Parameters
    ----------
    ratings
        Interactions to split.
    test_size
        Fraction of rows to place in the test set, in the open interval (0, 1).
    random_state
        Seed for reproducible splits.

    Returns
    -------
    tuple of pandas.DataFrame
        ``(train, test)`` with disjoint rows whose union is ``ratings``.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    n = len(ratings)
    rng = np.random.default_rng(random_state)
    order = rng.permutation(n)
    n_test = round(n * test_size)
    test_rows = np.sort(order[:n_test])
    train_rows = np.sort(order[n_test:])
    train = ratings.iloc[train_rows].reset_index(drop=True)
    test = ratings.iloc[test_rows].reset_index(drop=True)
    return train, test
