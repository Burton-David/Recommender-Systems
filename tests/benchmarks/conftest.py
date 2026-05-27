"""Fixtures for the algorithm benchmark suite.

The benchmarks live in ``tests/benchmarks/`` and are excluded from the default
``pytest`` run by the ``not benchmark`` marker filter in ``pyproject.toml``.
Run them with ``pytest tests/benchmarks/ -m benchmark`` (or
``pytest --benchmark-only``).
"""

from __future__ import annotations

import pandas as pd
import pytest

from recommender_systems.datasets import load_movielens_100k


@pytest.fixture(scope="session")
def movielens_100k() -> pd.DataFrame:
    """The full MovieLens 100k ratings frame.

    Cached at session scope because dataset I/O dominates wall time for the
    fast algorithms (MostPopular, MeanRating) and we don't want to re-read
    the file between benchmarks.
    """
    return load_movielens_100k()
