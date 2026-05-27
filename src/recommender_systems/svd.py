"""Matrix-factorization recommender via truncated SVD."""

from __future__ import annotations

from collections.abc import Hashable

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from recommender_systems.base import _SparseMatrixBackedRecommender
from recommender_systems.data import build_sparse_user_item_matrix

__all__ = ["SVD"]


class SVD(_SparseMatrixBackedRecommender):
    """Recommend from a low-rank reconstruction of the user-item matrix.

    Operates on the sparse user-item matrix directly. ``TruncatedSVD`` accepts
    CSR natively, so the dense ``(n_users, n_items)`` matrix never has to be
    materialized — the win that opens the goodbooks-full corpus to SVD.

    Parameters
    ----------
    n_factors
        Number of latent factors. Clamped to the matrix rank when smaller.
    random_state
        Seed for the SVD solver.
    """

    def __init__(self, n_factors: int = 20, random_state: int | None = None) -> None:
        super().__init__()
        self.n_factors = n_factors
        self.random_state = random_state
        self._user_factors: np.ndarray = np.empty((0, 0))
        self._item_factors: np.ndarray = np.empty((0, 0))

    def fit(self, ratings: pd.DataFrame) -> SVD:
        self._matrix, self._users, self._items = build_sparse_user_item_matrix(ratings)
        k = max(1, min(self.n_factors, min(self._matrix.shape) - 1))
        model = TruncatedSVD(n_components=k, random_state=self.random_state)
        self._user_factors = model.fit_transform(self._matrix)
        self._item_factors = model.components_
        return self

    def _user_scores(self, user_id: Hashable) -> np.ndarray:
        u_idx = self._users.get_loc(user_id)
        return np.asarray(self._user_factors[u_idx] @ self._item_factors)
