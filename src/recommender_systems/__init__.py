"""Recommender Systems: classic and modern recommendation algorithms."""

from recommender_systems.base import Recommender
from recommender_systems.data import build_user_item_matrix, split_interactions

__version__ = "0.1.0"

__all__ = ["Recommender", "build_user_item_matrix", "split_interactions"]
