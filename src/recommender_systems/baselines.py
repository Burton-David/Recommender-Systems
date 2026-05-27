"""Non-personalized baseline recommenders."""

from __future__ import annotations

from collections.abc import Hashable

import pandas as pd

from recommender_systems.base import Recommender

__all__ = ["MeanRating", "MostPopular"]


class _RankingBaseline(Recommender):
    """Recommend items by a global score, skipping ones the user has already rated."""

    def __init__(self) -> None:
        self._ranking: list[Hashable] = []
        self._seen: dict[Hashable, set[Hashable]] = {}

    def _rank(self, ratings: pd.DataFrame, scores: pd.Series) -> None:
        self._ranking = scores.sort_values(ascending=False).index.tolist()
        self._seen = ratings.groupby("user_id")["item_id"].agg(set).to_dict()

    def recommend(self, user_id: Hashable, n: int = 10) -> list[Hashable]:
        seen = self._seen.get(user_id, set())
        return [item for item in self._ranking if item not in seen][:n]


class MostPopular(_RankingBaseline):
    """Rank items by how often they have been rated."""

    def fit(self, ratings: pd.DataFrame) -> MostPopular:
        self._rank(ratings, ratings.groupby("item_id").size())
        return self


class MeanRating(_RankingBaseline):
    """Rank items by mean rating, requiring a minimum number of ratings.

    Parameters
    ----------
    min_ratings
        Minimum number of ratings an item needs to be eligible.
    """

    def __init__(self, min_ratings: int = 1) -> None:
        super().__init__()
        self.min_ratings = min_ratings

    def fit(self, ratings: pd.DataFrame) -> MeanRating:
        grouped = ratings.groupby("item_id")["rating"]
        means, counts = grouped.mean(), grouped.size()
        self._rank(ratings, means[counts >= self.min_ratings])
        return self
