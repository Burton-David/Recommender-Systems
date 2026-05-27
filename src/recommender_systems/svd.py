"""Matrix-factorization recommender via truncated SVD."""

from __future__ import annotations

from collections.abc import Hashable

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from recommender_systems.base import Recommender
from recommender_systems.data import build_user_item_matrix

__all__ = ["SVD"]


class SVD(Recommender):
    """Recommend from a low-rank reconstruction of the user-item matrix.

    Parameters
    ----------
    n_factors
        Number of latent factors. Clamped to the matrix rank when smaller.
    random_state
        Seed for the SVD solver.
    """

    def __init__(self, n_factors: int = 20, random_state: int | None = None) -> None:
        self.n_factors = n_factors
        self.random_state = random_state
        self._matrix = pd.DataFrame()
        self._predicted = np.empty((0, 0))

    def fit(self, ratings: pd.DataFrame) -> SVD:
        self._matrix = build_user_item_matrix(ratings, fill_value=0.0)
        k = max(1, min(self.n_factors, min(self._matrix.shape) - 1))
        model = TruncatedSVD(n_components=k, random_state=self.random_state)
        factors = model.fit_transform(self._matrix.to_numpy())
        self._predicted = factors @ model.components_
        return self

    def recommend(self, user_id: Hashable, n: int = 10) -> list[Hashable]:
        if user_id not in self._matrix.index:
            return []
        scores = self._predicted[self._matrix.index.get_loc(user_id)].copy()
        scores[self._matrix.loc[user_id].to_numpy() > 0] = -np.inf
        order = np.argsort(-scores)
        return [self._matrix.columns[i] for i in order if np.isfinite(scores[i])][:n]
