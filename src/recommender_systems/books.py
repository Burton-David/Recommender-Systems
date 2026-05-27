"""Book-specific helpers built on the goodbooks-10k loaders.

Drop-in pipelines for the book showcase — turn the tag table from
``load_goodbooks_tags`` into a content-based recommender in one call, so callers
don't have to repeat the join + vectorize + ``ContentBased`` boilerplate.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import pandas as pd

from recommender_systems.content import ContentBased
from recommender_systems.features import text_features

__all__ = ["build_tag_recommender", "tag_text_per_book"]


def tag_text_per_book(tags: pd.DataFrame) -> dict[Hashable, str]:
    """Concatenate each book's tag names into a single space-separated string.

    The schema matches what ``recommender_systems.datasets.load_goodbooks_tags``
    returns: columns ``book_id``, ``tag_name``, ``count``. Books that share many
    tags end up with overlapping vocabularies, which is what drives TF-IDF
    similarity downstream.
    """
    grouped: dict[Hashable, str] = tags.groupby("book_id")["tag_name"].agg(" ".join).to_dict()
    return grouped


def build_tag_recommender(tags: pd.DataFrame, **vectorizer_kwargs: Any) -> ContentBased:
    """Build a ``ContentBased`` recommender from goodbooks-style tag data.

    Each book's tags are joined into one document; ``text_features`` runs TF-IDF
    over the corpus to weigh rare-across-catalog tags higher than common ones.
    The returned recommender follows the side-information convention — fit it on
    the ratings frame (with ``item_id`` matching the ``book_id`` values used here)
    and call ``recommend(user_id, n)`` as usual.

    Parameters
    ----------
    tags
        DataFrame with columns ``book_id``, ``tag_name``, ``count``.
    **vectorizer_kwargs
        Forwarded to ``text_features`` (and thence to scikit-learn's
        ``TfidfVectorizer``) — e.g. ``stop_words="english"`` or ``min_df=2``.
    """
    features = text_features(tag_text_per_book(tags), method="tfidf", **vectorizer_kwargs)
    return ContentBased(item_features=features)
