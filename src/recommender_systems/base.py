"""The common interface and shared bases for recommenders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable

import numpy as np
import pandas as pd

__all__ = ["MatrixRecommender", "Recommender"]


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


class MatrixRecommender(Recommender):
    """Base for recommenders backed by a dense user-item matrix.

    Subclasses build ``self._matrix`` in :meth:`fit` and implement :meth:`_score_items`;
    :meth:`recommend` then ranks the items a user has not yet rated by that score.
    """

    def __init__(self) -> None:
        self._matrix = pd.DataFrame()

    def recommend(self, user_id: Hashable, n: int = 10) -> list[Hashable]:
        if user_id not in self._matrix.index:
            return []
        scores = np.array(self._score_items(user_id), dtype=float)
        scores[self._matrix.loc[user_id].to_numpy() > 0] = -np.inf
        order = np.argsort(-scores)
        return [self._matrix.columns[i] for i in order if np.isfinite(scores[i])][:n]

    @abstractmethod
    def _score_items(self, user_id: Hashable) -> np.ndarray:
        """Return a score for every item, aligned with ``self._matrix.columns``."""
