"""Neighborhood (k-NN) collaborative filtering on sparse matrices."""

from __future__ import annotations

from collections.abc import Hashable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from recommender_systems.base import _SparseMatrixBackedRecommender
from recommender_systems.data import build_sparse_user_item_matrix

__all__ = ["ItemKNN", "UserKNN"]


class ItemKNN(_SparseMatrixBackedRecommender):
    """Score items by similarity to the items a user has already rated.

    Builds a sparse item-item cosine similarity matrix, keeps the ``k`` largest
    entries per row, and scores by ``S @ user_row``. The full similarity is
    materialized in CSR form, which is fine even for goodbooks-10k (10k by 10k
    items would be 100M entries dense; in sparse form after the top-``k`` trim
    it's `k * n_items` = a few hundred thousand non-zeros).

    Parameters
    ----------
    k
        Number of nearest neighbors retained per item.
    """

    def __init__(self, k: int = 20) -> None:
        super().__init__()
        self.k = k
        self._similarity: sparse.csr_matrix = sparse.csr_matrix((0, 0))

    def fit(self, ratings: pd.DataFrame) -> ItemKNN:
        self._matrix, self._users, self._items = build_sparse_user_item_matrix(ratings)
        full = cosine_similarity(self._matrix.T, dense_output=False)
        self._similarity = _sparse_top_k_per_row(full.tocsr(), self.k)
        return self

    def _user_scores(self, user_id: Hashable) -> np.ndarray:
        u_idx = self._users.get_loc(user_id)
        user_row = self._matrix.getrow(u_idx)
        scores = self._similarity @ user_row.T
        return np.asarray(scores.todense()).ravel()


class UserKNN(_SparseMatrixBackedRecommender):
    """Score items by the ratings of a user's nearest neighbors.

    Builds the neighborhood with :class:`sklearn.neighbors.NearestNeighbors`
    (cosine metric on the sparse user-item matrix) rather than materializing
    the full users-by-users similarity — that's ``n_users**2`` and blows up
    past a few thousand users. Per-user scoring weights the neighbors' rating
    rows by their similarities.

    Parameters
    ----------
    k
        Number of nearest neighbors retained per user.
    """

    def __init__(self, k: int = 20) -> None:
        super().__init__()
        self.k = k
        self._neighbor_idx: np.ndarray = np.empty((0, 0), dtype=np.int64)
        self._neighbor_sim: np.ndarray = np.empty((0, 0))

    def fit(self, ratings: pd.DataFrame) -> UserKNN:
        self._matrix, self._users, self._items = build_sparse_user_item_matrix(ratings)
        # Ask for k+1 neighbors because each user is its own first neighbor
        # (cosine distance 0); we drop that column below.
        n_neighbors = min(self.k + 1, self._matrix.shape[0])
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
        nn.fit(self._matrix)
        distances, indices = nn.kneighbors(self._matrix)
        self._neighbor_idx = indices[:, 1:]
        self._neighbor_sim = 1.0 - distances[:, 1:]
        return self

    def _user_scores(self, user_id: Hashable) -> np.ndarray:
        u_idx = self._users.get_loc(user_id)
        neighbors = self._neighbor_idx[u_idx]
        weights = self._neighbor_sim[u_idx]
        if not weights.size or weights.sum() <= 0:
            return np.zeros(self._matrix.shape[1])
        neighbor_rows = self._matrix[neighbors]
        scores = weights @ neighbor_rows
        return np.asarray(scores.todense() if sparse.issparse(scores) else scores).ravel() / (
            weights.sum() or 1.0
        )


def _sparse_top_k_per_row(matrix: sparse.csr_matrix, k: int) -> sparse.csr_matrix:
    """Zero the diagonal and keep the ``k`` largest entries in each row."""
    matrix = matrix.copy()
    matrix.setdiag(0.0)
    matrix.eliminate_zeros()
    if k >= matrix.shape[1]:
        return matrix
    rows, cols, data = [], [], []
    for row_idx in range(matrix.shape[0]):
        start, end = matrix.indptr[row_idx], matrix.indptr[row_idx + 1]
        if end - start <= k:
            rows.extend([row_idx] * (end - start))
            cols.extend(matrix.indices[start:end])
            data.extend(matrix.data[start:end])
            continue
        row_data = matrix.data[start:end]
        row_cols = matrix.indices[start:end]
        # argpartition is O(n) where n = number of non-zeros in this row.
        top_local = np.argpartition(row_data, -k)[-k:]
        rows.extend([row_idx] * k)
        cols.extend(row_cols[top_local])
        data.extend(row_data[top_local])
    return sparse.csr_matrix((data, (rows, cols)), shape=matrix.shape)
