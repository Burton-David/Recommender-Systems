"""Recommender Systems: classic and modern recommendation algorithms."""

from recommender_systems.base import Recommender
from recommender_systems.baselines import MeanRating, MostPopular
from recommender_systems.bpr import BPR
from recommender_systems.content import ContentBased
from recommender_systems.data import (
    build_user_item_matrix,
    densest_subset,
    holdout_per_user,
    split_ratings,
)
from recommender_systems.hybrid import HybridRecommender
from recommender_systems.neighborhood import ItemKNN, UserKNN
from recommender_systems.svd import SVD

__version__ = "0.1.0"

__all__ = [
    "BPR",
    "SVD",
    "ContentBased",
    "HybridRecommender",
    "ItemKNN",
    "MeanRating",
    "MostPopular",
    "Recommender",
    "UserKNN",
    "build_user_item_matrix",
    "densest_subset",
    "holdout_per_user",
    "split_ratings",
]
