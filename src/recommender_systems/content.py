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
        self._predicted = np.empty((0, 0))

    def fit(self, ratings: pd.DataFrame) -> ContentBased:
        rated_matrix = build_user_item_matrix(ratings, fill_value=0.0)
        # Items with features but no ratings still need to be recommendable —
        # that's the whole point of content-based — so widen the matrix to the
        # union of rated and featured items.
        item_space = rated_matrix.columns.union(self._item_features.index)
        self._matrix = rated_matrix.reindex(columns=item_space, fill_value=0.0)
        features = self._item_features.reindex(self._matrix.columns).fillna(0.0).to_numpy()
        weights = self._matrix.to_numpy()
        row_sums = weights.sum(axis=1, keepdims=True)
        profiles = np.divide(weights @ features, np.where(row_sums == 0, 1.0, row_sums))
        self._predicted = np.asarray(cosine_similarity(profiles, features))
        return self

    def _user_scores(self, user_id: Hashable) -> np.ndarray:
        return np.asarray(self._predicted[self._matrix.index.get_loc(user_id)])
