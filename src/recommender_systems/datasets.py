"""Dataset loaders for common recommender benchmarks."""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

__all__ = [
    "load_goodbooks_10k",
    "load_goodbooks_books",
    "load_goodbooks_tags",
    "load_movielens_100k",
]

_DEFAULT_HOME = Path.home() / ".recommender_systems"
_ML_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
_ML_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]
_GOODBOOKS_BASE = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master"


def load_movielens_100k(data_home: str | Path | None = None) -> pd.DataFrame:
    """Load the MovieLens 100k ratings, downloading and caching on first use.

    Parameters
    ----------
    data_home
        Directory to cache the dataset in. Defaults to ``~/.recommender_systems``.

    Returns
    -------
    pandas.DataFrame
        Ratings with columns ``user_id``, ``item_id``, ``rating``, ``timestamp``.
    """
    home = Path(data_home) if data_home is not None else _DEFAULT_HOME
    data_file = home / "ml-100k" / "u.data"
    if not data_file.exists():
        home.mkdir(parents=True, exist_ok=True)
        archive = home / "ml-100k.zip"
        urllib.request.urlretrieve(_ML_100K_URL, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(home)
    return pd.read_csv(data_file, sep="\t", names=_ML_COLUMNS)


def load_goodbooks_10k(data_home: str | Path | None = None) -> pd.DataFrame:
    """Load the goodbooks-10k ratings, downloading and caching on first use.

    For the open-source benchmark/demo only: the dataset derives from Goodreads and is
    not licensed for shipping in a commercial app (see the README).

    Returns
    -------
    pandas.DataFrame
        Ratings with columns ``user_id``, ``book_id``, ``rating``.
    """
    return pd.read_csv(_goodbooks_file(data_home, "ratings.csv"))


def load_goodbooks_books(data_home: str | Path | None = None) -> pd.DataFrame:
    """Load goodbooks-10k book metadata (title, authors, ids, average rating, ...)."""
    return pd.read_csv(_goodbooks_file(data_home, "books.csv"))


def load_goodbooks_tags(data_home: str | Path | None = None) -> pd.DataFrame:
    """Load goodbooks-10k tags joined to ``book_id`` and tag name.

    Returns
    -------
    pandas.DataFrame
        Columns ``book_id``, ``tag_name``, ``count`` — ready to build content features.
    """
    book_tags = pd.read_csv(_goodbooks_file(data_home, "book_tags.csv"))
    tags = pd.read_csv(_goodbooks_file(data_home, "tags.csv"))
    ids = pd.read_csv(
        _goodbooks_file(data_home, "books.csv"), usecols=["book_id", "goodreads_book_id"]
    )
    merged = book_tags.merge(tags, on="tag_id").merge(ids, on="goodreads_book_id")
    return merged[["book_id", "tag_name", "count"]]


def _goodbooks_file(data_home: str | Path | None, name: str) -> Path:
    home = Path(data_home) if data_home is not None else _DEFAULT_HOME
    path = home / "goodbooks-10k" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{_GOODBOOKS_BASE}/{name}", path)
    return path
