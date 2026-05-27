"""Matrix-factorization recommender via truncated SVD."""

from __future__ import annotations

from collections.abc import Hashable

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from recommender_systems.base import _MatrixBackedRecommender
from recommender_systems.data import build_user_item_matrix

__all__ = ["SVD"]


class SVD(_MatrixBackedRecommender):
    """Recommend from a low-rank reconstruction of the user-item matrix.

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
        self._predicted = np.empty((0, 0))

    def fit(self, ratings: pd.DataFrame) -> SVD:
        self._matrix = build_user_item_matrix(ratings, fill_value=0.0)
        k = max(1, min(self.n_factors, min(self._matrix.shape) - 1))
        model = TruncatedSVD(n_components=k, random_state=self.random_state)
        factors = model.fit_transform(self._matrix.to_numpy())
        self._predicted = factors @ model.components_
        return self

    def _user_scores(self, user_id: Hashable) -> np.ndarray:
        return np.asarray(self._predicted[self._matrix.index.get_loc(user_id)])
