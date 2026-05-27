"""Book-specific helpers built on the goodbooks-10k loaders.

Drop-in pipelines for the book recommender — turn the tag table from
``load_goodbooks_tags`` into a content-based recommender in one call, so callers
don't have to repeat the join + vectorize + ``ContentBased`` boilerplate.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import pandas as pd

from recommender_systems.base import Recommender
from recommender_systems.content import ContentBased
from recommender_systems.features import text_features
from recommender_systems.hybrid import HybridRecommender
from recommender_systems.neighborhood import ItemKNN

__all__ = ["build_hybrid_book_recommender", "build_tag_recommender", "tag_text_per_book"]


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


def build_hybrid_book_recommender(
    tags: pd.DataFrame,
    *,
    collaborative: Recommender | None = None,
    weights: tuple[float, float] = (3.0, 1.0),
    rank_constant: int = 60,
    **vectorizer_kwargs: Any,
) -> HybridRecommender:
    """Blend a collaborative recommender with the tag-based content recommender.

    Defaults to ``ItemKNN(k=20)`` on the collaborative side because item-item kNN
    composes well with content signal (both rank items by similarity, just over
    different spaces). Pass any other ``Recommender`` to swap that out.

    The default ``weights=(3.0, 1.0)`` puts the collaborative signal in charge —
    on benchmarked datasets the dense CF signal is much stronger than the
    tag-only content one, so equal weighting dilutes accuracy. The content half
    still contributes useful boosts for items both signals agree on, plus a
    cold-start fallback for items the CF half has never seen.

    Parameters
    ----------
    tags
        DataFrame with columns ``book_id``, ``tag_name``, ``count`` — fed into
        :func:`build_tag_recommender`.
    collaborative
        Recommender to use on the collaborative side. ``None`` (default) uses
        ``ItemKNN(k=20)``.
    weights
        ``(collab_weight, content_weight)`` for the underlying
        :class:`HybridRecommender` (RRF fusion).
    rank_constant
        Forwarded to :class:`HybridRecommender`.
    **vectorizer_kwargs
        Forwarded to :func:`build_tag_recommender` (and thence to scikit-learn's
        ``TfidfVectorizer``) — useful for ``max_features`` on big tag catalogs.
    """
    content = build_tag_recommender(tags, **vectorizer_kwargs)
    collab = collaborative if collaborative is not None else ItemKNN(k=20)
    return HybridRecommender([collab, content], weights=list(weights), rank_constant=rank_constant)
