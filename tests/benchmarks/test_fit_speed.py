"""Benchmark how long each algorithm takes to fit MovieLens 100k.

Run with::

    pytest tests/benchmarks/ -m benchmark --benchmark-only --benchmark-columns=mean,stddev,rounds

The committed baseline lives in ``benchmarks/profile.md``. CI does not run
benchmarks (they are wall-time-sensitive and noisy on shared hardware); the
suite exists so any local rewrite has a concrete number to beat.
"""

from __future__ import annotations

import pandas as pd
import pytest

from recommender_systems.als import ALS
from recommender_systems.baselines import MeanRating, MostPopular
from recommender_systems.bpr import BPR
from recommender_systems.neighborhood import ItemKNN, UserKNN
from recommender_systems.svd import SVD

pytestmark = pytest.mark.benchmark


def test_most_popular_fit(benchmark, movielens_100k: pd.DataFrame) -> None:
    benchmark(lambda: MostPopular().fit(movielens_100k))


def test_mean_rating_fit(benchmark, movielens_100k: pd.DataFrame) -> None:
    benchmark(lambda: MeanRating().fit(movielens_100k))


def test_item_knn_fit(benchmark, movielens_100k: pd.DataFrame) -> None:
    benchmark(lambda: ItemKNN().fit(movielens_100k))


def test_user_knn_fit(benchmark, movielens_100k: pd.DataFrame) -> None:
    benchmark(lambda: UserKNN().fit(movielens_100k))


def test_svd_fit(benchmark, movielens_100k: pd.DataFrame) -> None:
    benchmark(lambda: SVD(n_factors=20, random_state=0).fit(movielens_100k))


def test_bpr_fit(benchmark, movielens_100k: pd.DataFrame) -> None:
    # Five epochs keeps the benchmark under ~30s on a laptop while still
    # exercising the full inner loop.
    benchmark(lambda: BPR(n_factors=16, epochs=5, random_state=0).fit(movielens_100k))


def test_als_fit(benchmark, movielens_100k: pd.DataFrame) -> None:
    benchmark(lambda: ALS(n_factors=16, epochs=5, random_state=0).fit(movielens_100k))
