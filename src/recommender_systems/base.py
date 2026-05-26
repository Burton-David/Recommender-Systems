"""The common interface shared by all recommenders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
