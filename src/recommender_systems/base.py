"""The common interface shared by all recommenders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable

import numpy as np
import pandas as pd

__all__ = ["Recommender"]


class Recommender(ABC):
    """Abstract base class for recommendation algorithms.

    Concrete recommenders learn from interaction data in :meth:`fit` and produce a ranked
    list of item ids in :meth:`recommend`. Sharing one interface lets algorithms be swapped
    and evaluated interchangeably.
    """

    @abstractmethod
    def fit(self, ratings: pd.DataFrame) -> Recommender:
        """Train on a ratings frame.

        Parameters
        ----------
        ratings:
            Interaction data with at least ``user_id``, ``item_id`` and ``rating`` columns.

        Returns
        -------
        Recommender
            The fitted instance, so calls can be chained.
        """

    @abstractmethod
    def recommend(self, user_id: Hashable, n: int = 10) -> list[Hashable]:
        """Return the top-``n`` recommended item ids for ``user_id``.

        Parameters
        ----------
        user_id:
            The user to recommend for.
        n:
            Maximum number of items to return.

        Returns
        -------
        list of Hashable
            Item ids ordered from most to least recommended.
        """


class _MatrixBackedRecommender(Recommender):
    """Recommend by ranking the user's row of a materialized user-item matrix.

    Subclasses fill in :meth:`fit` (which must set ``self._matrix``) and
    :meth:`_user_scores`, an item-aligned numpy vector. This base owns the
    unknown-user fast path, the seen-item mask, and the top-``n`` filter, so
    every matrix-backed algorithm shares one recommend contract.
    """

    def __init__(self) -> None:
        self._matrix = pd.DataFrame()

    @abstractmethod
    def _user_scores(self, user_id: Hashable) -> np.ndarray:
        """Return an item-aligned score vector for ``user_id``."""

    def recommend(self, user_id: Hashable, n: int = 10) -> list[Hashable]:
        if user_id not in self._matrix.index:
            return []
        scores = self._user_scores(user_id).copy()
        scores[self._matrix.loc[user_id].to_numpy() > 0] = -np.inf
        order = np.argsort(-scores)
        return [self._matrix.columns[i] for i in order if np.isfinite(scores[i])][:n]


class _PredictedScoreRecommender(_MatrixBackedRecommender):
    """Matrix-backed recommender that precomputes a dense user-item score matrix.

    Subclasses populate ``self._predicted`` (users x items, row-aligned with
    ``self._matrix``) during :meth:`fit`; scoring a user is then a row lookup.
    """

    def __init__(self) -> None:
        super().__init__()
        self._predicted: np.ndarray = np.empty((0, 0))

    def _user_scores(self, user_id: Hashable) -> np.ndarray:
        return np.asarray(self._predicted[self._matrix.index.get_loc(user_id)])
