"""Build item feature matrices from text, for use with :class:`~recommender_systems.content.ContentBased`."""

from __future__ import annotations

from collections.abc import Hashable, Mapping

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

__all__ = ["text_features"]


def text_features(
    item_text: Mapping[Hashable, str], *, method: str = "tfidf", **vectorizer_kwargs: object
) -> pd.DataFrame:
    """Vectorize each item's text into a numerical feature matrix.

    The result is indexed by item id with one column per term, ready to pass to
    ``ContentBased(item_features=...)``. Item descriptions, concatenated tags, or any
    per-item text all work.

    Parameters
    ----------
    item_text
        Mapping from item id to its text.
    method
        ``"tfidf"`` (TF-IDF weights), ``"count"`` (term counts), or ``"binary"``
        (term presence).
    **vectorizer_kwargs
        Forwarded to the underlying scikit-learn vectorizer (e.g. ``stop_words``,
        ``max_features``, ``ngram_range``).

    Returns
    -------
    pandas.DataFrame
        Item-by-term feature matrix.
    """
    items = list(item_text)
    corpus = [item_text[item] for item in items]
    if method == "tfidf":
        vectorizer = TfidfVectorizer(**vectorizer_kwargs)
    elif method == "count":
        vectorizer = CountVectorizer(**vectorizer_kwargs)
    elif method == "binary":
        vectorizer = CountVectorizer(binary=True, **vectorizer_kwargs)
    else:
        raise ValueError(f"unknown method {method!r}; use 'tfidf', 'count', or 'binary'")
    matrix = vectorizer.fit_transform(corpus).toarray()
    return pd.DataFrame(matrix, index=items, columns=vectorizer.get_feature_names_out())
