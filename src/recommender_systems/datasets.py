"""Dataset loaders for common recommender benchmarks."""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

__all__ = ["load_movielens_100k"]

_ML_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
_DEFAULT_HOME = Path.home() / ".recommender_systems"
_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]


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
        _download_and_extract(home)
    return pd.read_csv(data_file, sep="\t", names=_COLUMNS)


def _download_and_extract(home: Path) -> None:
    home.mkdir(parents=True, exist_ok=True)
    archive = home / "ml-100k.zip"
    urllib.request.urlretrieve(_ML_100K_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(home)
