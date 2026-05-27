"""Utilities for preparing interaction data for recommenders."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

__all__ = [
    "build_sparse_user_item_matrix",
    "build_user_item_matrix",
    "densest_subset",
    "holdout_per_user",
    "split_ratings",
]


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


def split_ratings(
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


def densest_subset(
    ratings: pd.DataFrame,
    *,
    n_users: int = 1000,
    n_items: int = 1000,
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> pd.DataFrame:
    """Restrict ratings to the most active users and most popular items.

    Lets dense-matrix algorithms run on large datasets (e.g. goodbooks-10k) without
    materializing a full user-by-item matrix. Keeps the ``n_users`` users with the most
    interactions and the ``n_items`` most-interacted items, then the rows in both.

    Returns
    -------
    pandas.DataFrame
        The filtered ratings.
    """
    top_users = ratings[user_col].value_counts().head(n_users).index
    top_items = ratings[item_col].value_counts().head(n_items).index
    keep = ratings[user_col].isin(top_users) & ratings[item_col].isin(top_items)
    return ratings[keep].reset_index(drop=True)


def holdout_per_user(
    ratings: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int | None = None,
    user_col: str = "user_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out a fraction of each user's interactions for testing.

    Unlike :func:`split_ratings` (a global row split), this guarantees every user with at
    least two interactions keeps training history — the standard protocol for top-N
    recommender evaluation, so no user is left cold in the test set. Users with a single
    interaction stay entirely in train.

    Returns
    -------
    tuple of pandas.DataFrame
        ``(train, test)``.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    rng = np.random.default_rng(random_state)
    test_mask = np.zeros(len(ratings), dtype=bool)
    for positions in ratings.groupby(user_col).indices.values():
        if len(positions) < 2:
            continue
        n_test = min(max(1, round(len(positions) * test_size)), len(positions) - 1)
        test_mask[rng.choice(positions, size=n_test, replace=False)] = True
    train = ratings[~test_mask].reset_index(drop=True)
    test = ratings[test_mask].reset_index(drop=True)
    return train, test


def build_sparse_user_item_matrix(
    ratings: pd.DataFrame,
    *,
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: str = "rating",
) -> tuple[sparse.csr_matrix, pd.Index, pd.Index]:
    """Build a sparse user-item matrix plus its id-to-position index maps.

    Scales to corpora where the dense :func:`build_user_item_matrix` would not fit in
    memory (e.g. full goodbooks-10k). Duplicate (user, item) pairs are averaged, matching
    the dense builder.

    Returns
    -------
    tuple
        ``(matrix, users, items)`` — a CSR matrix, an index mapping row to user id, and an
        index mapping column to item id (use ``.get_loc(id)`` for the reverse direction).
    """
    missing = {user_col, item_col, rating_col} - set(ratings.columns)
    if missing:
        raise ValueError(f"ratings is missing required columns: {sorted(missing)}")
    agg = ratings.groupby([user_col, item_col], sort=True)[rating_col].mean().reset_index()
    user_codes, users = pd.factorize(agg[user_col], sort=True)
    item_codes, items = pd.factorize(agg[item_col], sort=True)
    matrix = sparse.csr_matrix(
        (agg[rating_col].to_numpy(), (user_codes, item_codes)),
        shape=(len(users), len(items)),
    )
    return matrix, pd.Index(users), pd.Index(items)
