"""Benchmark every top-level recommender on MovieLens 100k.

Regenerates ``benchmarks/results.md`` and ``benchmarks/results.png``.

Usage::

    pip install -e ".[dev,benchmarks]"
    python -m scripts.benchmark
"""

from __future__ import annotations

from pathlib import Path

from recommender_systems import SVD, ItemKNN, MeanRating, MostPopular, Recommender, UserKNN
from recommender_systems.datasets import load_movielens_100k
from scripts._harness import run_benchmark

SEED = 0
OUT_DIR = Path(__file__).resolve().parent.parent / "benchmarks"


def build_models() -> dict[str, Recommender]:
    return {
        "MostPopular": MostPopular(),
        "MeanRating": MeanRating(min_ratings=5),
        "ItemKNN": ItemKNN(k=20),
        "UserKNN": UserKNN(k=20),
        "SVD": SVD(n_factors=50, random_state=SEED),
    }


def main() -> None:
    print("Loading MovieLens 100k ...")
    ratings = load_movielens_100k()
    run_benchmark(
        "MovieLens 100k",
        ratings,
        build_models(),
        out_dir=OUT_DIR,
        file_stem="results",
        script="python -m scripts.benchmark",
        seed=SEED,
    )


if __name__ == "__main__":
    main()
