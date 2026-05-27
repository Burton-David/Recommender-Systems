"""Benchmark every top-level recommender on goodbooks-10k.

Regenerates ``benchmarks/goodbooks_results.md`` and ``benchmarks/goodbooks_results.png``.

Usage::

    pip install -e ".[dev,benchmarks]"
    python -m scripts.benchmark_goodbooks

The full goodbooks-10k corpus has ~53k users and ~6M interactions; a dense user-user
similarity at that scale would need ~22 GB of memory. The benchmark subsamples to a
deterministic seed-picked 2500-user slice (~280k interactions) so every algorithm —
including UserKNN — fits in memory and runs in a couple of minutes on a laptop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from recommender_systems import SVD, ItemKNN, MeanRating, MostPopular, Recommender, UserKNN
from recommender_systems.books import build_hybrid_book_recommender
from recommender_systems.datasets import load_goodbooks_10k, load_goodbooks_tags
from scripts._harness import run_benchmark

SEED = 20260527
USER_SAMPLE = 2500
OUT_DIR = Path(__file__).resolve().parent.parent / "benchmarks"


def build_models(tags: pd.DataFrame) -> dict[str, Recommender]:
    return {
        "MostPopular": MostPopular(),
        "MeanRating": MeanRating(min_ratings=20),
        "ItemKNN": ItemKNN(k=20),
        "UserKNN": UserKNN(k=20),
        "SVD": SVD(n_factors=50, random_state=SEED),
        # Cap the tag vocabulary so the dense item-feature matrix stays small.
        "HybridBook": build_hybrid_book_recommender(tags, max_features=200),
    }


def main() -> None:
    print("Loading goodbooks-10k ...")
    ratings = load_goodbooks_10k().rename(columns={"book_id": "item_id"})
    tags = load_goodbooks_tags()

    rng = np.random.default_rng(SEED)
    all_users = ratings["user_id"].unique()
    sampled = rng.choice(all_users, size=min(USER_SAMPLE, len(all_users)), replace=False)
    ratings = ratings[ratings["user_id"].isin(sampled)].reset_index(drop=True)
    print(f"  sampled {len(sampled)} users, {len(ratings):,} interactions")

    run_benchmark(
        "goodbooks-10k",
        ratings,
        build_models(tags),
        out_dir=OUT_DIR,
        file_stem="goodbooks_results",
        script="python -m scripts.benchmark_goodbooks",
        seed=SEED,
        notes=(
            f"Subsampled to {USER_SAMPLE} users (~{len(ratings) // 1000}k interactions) so "
            "the dense user-user similarity needed by UserKNN fits in memory. "
            "HybridBook fuses ItemKNN with a tag-based ContentBased via "
            "HybridRecommender (RRF)."
        ),
    )


if __name__ == "__main__":
    main()
