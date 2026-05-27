"""Content-based recommendation by item-feature similarity."""

from __future__ import annotations

from collections.abc import Hashable

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from recommender_systems.base import _MatrixBackedRecommender
from recommender_systems.data import build_user_item_matrix

__all__ = ["ContentBased"]


class ContentBased(_MatrixBackedRecommender):
    """Recommend items whose features resemble those a user has liked.

    The user's profile is a ratings-weighted mean of the features of the items they
    rated; recommendations are ranked by cosine similarity between that profile and
    each candidate item.

    Side information is passed at **construction time** rather than to ``fit``. This
    keeps ``fit(ratings)`` uniform across the library so algorithms stay interchangeable;
    see ``CONTRIBUTING.md`` for the convention.

    Parameters
    ----------
    item_features
        DataFrame indexed by item id with one numerical feature per column.
        Callers pre-compute features from raw inputs (TF-IDF over descriptions,
        embeddings, one-hot categoricals, etc.) before passing them in.
    """

    def __init__(self, item_features: pd.DataFrame) -> None:
        super().__init__()
        self._item_features = item_features
        self._predicted: np.ndarray = np.empty((0, 0))
        self._aligned_features: np.ndarray = np.empty((0, 0))
        self._profiles: np.ndarray = np.empty((0, 0))
        self._feature_names: list[str] = []

    def fit(self, ratings: pd.DataFrame) -> ContentBased:
        rated_matrix = build_user_item_matrix(ratings, fill_value=0.0)
        # Items with features but no ratings still need to be recommendable —
        # that's the whole point of content-based — so widen the matrix to the
        # union of rated and featured items.
        item_space = rated_matrix.columns.union(self._item_features.index)
        self._matrix = rated_matrix.reindex(columns=item_space, fill_value=0.0)
        aligned = self._item_features.reindex(self._matrix.columns).fillna(0.0)
        self._aligned_features = aligned.to_numpy()
        self._feature_names = list(aligned.columns)
        weights = self._matrix.to_numpy()
        row_sums = weights.sum(axis=1, keepdims=True)
        self._profiles = np.divide(
            weights @ self._aligned_features, np.where(row_sums == 0, 1.0, row_sums)
        )
        self._predicted = np.asarray(cosine_similarity(self._profiles, self._aligned_features))
        return self

    def _user_scores(self, user_id: Hashable) -> np.ndarray:
        return np.asarray(self._predicted[self._matrix.index.get_loc(user_id)])

    def explain(self, user_id: Hashable, item_id: Hashable, top_features: int = 3) -> str:
        """Return the top features driving ``item_id``'s score for ``user_id``.

        For each feature, the contribution to the user-item similarity is the
        product of the user's profile weight and the item's feature weight.
        The top non-zero features are returned as a comma-separated string —
        e.g. ``"fantasy, magic, epic"`` for a fantasy-genre recommendation.
        Returns ``""`` if the user or item is unknown or the user has no profile.
        """
        if user_id not in self._matrix.index or item_id not in self._matrix.columns:
            return ""
        u_idx = self._matrix.index.get_loc(user_id)
        i_idx = self._matrix.columns.get_loc(item_id)
        contributions = self._profiles[u_idx] * self._aligned_features[i_idx]
        if not contributions.any():
            return ""
        top_idx = np.argsort(-contributions)[:top_features]
        return ", ".join(self._feature_names[i] for i in top_idx if contributions[i] > 0)

    def recommend_with_reasons(
        self, user_id: Hashable, n: int = 10, top_features: int = 3
    ) -> list[tuple[Hashable, str]]:
        """Top-``n`` recommendations for ``user_id`` paired with their explanations.

        Each entry is ``(item_id, reason)`` where ``reason`` is the output of
        :meth:`explain` for that item — empty string if no features contribute.
        """
        return [
            (item, self.explain(user_id, item, top_features)) for item in self.recommend(user_id, n)
        ]
