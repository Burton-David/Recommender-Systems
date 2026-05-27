"""Combine several recommenders by weighted reciprocal-rank fusion."""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import pandas as pd

from recommender_systems.base import Recommender

__all__ = ["HybridRecommender"]


class HybridRecommender(Recommender):
    """Blend several recommenders via weighted reciprocal-rank fusion.

    Each component produces its own ranked list; an item scores
    ``sum_r weight_r / (rank_constant + rank_r)``, so items ranked highly by several
    weighted components rise to the top. Because the components already exclude items
    the user has rated, the fused list does too. Rank fusion needs only the public
    ``recommend`` output, so any recommender can be combined.

    Parameters
    ----------
    recommenders
        Component recommenders to combine.
    weights
        Per-component weights; defaults to equal weighting.
    rank_constant
        Dampens the contribution of top ranks (the standard RRF constant).
    pool
        How many items to pull from each component before fusing.
    """

    def __init__(
        self,
        recommenders: Sequence[Recommender],
        weights: Sequence[float] | None = None,
        *,
        rank_constant: int = 60,
        pool: int = 100,
    ) -> None:
        if not recommenders:
            raise ValueError("need at least one recommender")
        if weights is not None and len(weights) != len(recommenders):
            raise ValueError("weights must match the number of recommenders")
        self.recommenders = list(recommenders)
        self.weights = list(weights) if weights is not None else [1.0] * len(recommenders)
        self.rank_constant = rank_constant
        self.pool = pool

    def fit(self, ratings: pd.DataFrame) -> HybridRecommender:
        for recommender in self.recommenders:
            recommender.fit(ratings)
        return self

    def recommend(self, user_id: Hashable, n: int = 10) -> list[Hashable]:
        scores: dict[Hashable, float] = {}
        for recommender, weight in zip(self.recommenders, self.weights, strict=True):
            for rank, item in enumerate(recommender.recommend(user_id, n=self.pool), start=1):
                scores[item] = scores.get(item, 0.0) + weight / (self.rank_constant + rank)
        ranked = sorted(scores, key=lambda item: scores[item], reverse=True)
        return ranked[:n]
