"""Neighborhood (k-NN) collaborative filtering."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Hashable

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from recommender_systems.base import Recommender
from recommender_systems.data import build_user_item_matrix

__all__ = ["ItemKNN", "UserKNN"]


def _keep_top_k(similarity: np.ndarray, k: int) -> np.ndarray:
    """Zero the diagonal and all but the ``k`` largest entries in each row."""
    similarity = similarity.copy()
    np.fill_diagonal(similarity, 0.0)
    if k < similarity.shape[1]:
        drop = np.argsort(-similarity, axis=1)[:, k:]
        np.put_along_axis(similarity, drop, 0.0, axis=1)
    return similarity


class _NeighborhoodCF(Recommender):
    """Shared fit/recommend for neighborhood collaborative filtering."""

    def __init__(self, k: int = 20) -> None:
        self.k = k
        self._matrix = pd.DataFrame()
        self._similarity = np.empty((0, 0))

    def fit(self, ratings: pd.DataFrame) -> _NeighborhoodCF:
        self._matrix = build_user_item_matrix(ratings, fill_value=0.0)
        self._similarity = _keep_top_k(self._similarity_of(self._matrix.to_numpy()), self.k)
        return self

    def recommend(self, user_id: Hashable, n: int = 10) -> list[Hashable]:
        if user_id not in self._matrix.index:
            return []
        scores = self._scores(user_id)
        scores[self._matrix.loc[user_id].to_numpy() > 0] = -np.inf
        order = np.argsort(-scores)
        return [self._matrix.columns[i] for i in order if np.isfinite(scores[i])][:n]

    @abstractmethod
    def _similarity_of(self, matrix: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def _scores(self, user_id: Hashable) -> np.ndarray: ...


class ItemKNN(_NeighborhoodCF):
    """Score items by similarity to the items a user has already rated."""

    def _similarity_of(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(cosine_similarity(matrix.T))

    def _scores(self, user_id: Hashable) -> np.ndarray:
        return np.asarray(self._similarity @ self._matrix.loc[user_id].to_numpy())


class UserKNN(_NeighborhoodCF):
    """Score items by the ratings of a user's nearest neighbors."""

    def _similarity_of(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(cosine_similarity(matrix))

    def _scores(self, user_id: Hashable) -> np.ndarray:
        weights = self._similarity[self._matrix.index.get_loc(user_id)]
        return np.asarray(weights @ self._matrix.to_numpy() / (weights.sum() or 1.0))
