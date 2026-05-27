"""Benchmark every top-level recommender on MovieLens 100k.

Regenerates ``benchmarks/results.md`` and ``benchmarks/results.png``.

Usage::

    pip install -e ".[dev,benchmarks]"
    python scripts/benchmark.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from recommender_systems import Recommender, split_ratings
from recommender_systems.baselines import MeanRating, MostPopular
from recommender_systems.datasets import load_movielens_100k
from recommender_systems.metrics import (
    catalog_coverage,
    mean_average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from recommender_systems.neighborhood import ItemKNN, UserKNN
from recommender_systems.svd import SVD

K = 10
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


def evaluate(
    model: Recommender,
    train: pd.DataFrame,
    test: pd.DataFrame,
    catalog: set[int],
) -> dict[str, float]:
    model.fit(train)
    truth = test.groupby("user_id")["item_id"].agg(set)
    users = sorted(truth.index)
    predicted = [model.recommend(u, n=K) for u in users]
    actual = [truth[u] for u in users]
    return {
        f"precision@{K}": precision_at_k(predicted, actual, K),
        f"recall@{K}": recall_at_k(predicted, actual, K),
        f"MAP@{K}": mean_average_precision(predicted, actual, K),
        f"NDCG@{K}": ndcg_at_k(predicted, actual, K),
        f"coverage@{K}": catalog_coverage(predicted, catalog, K),
    }


def render_table(results: pd.DataFrame) -> str:
    body = results.map(lambda v: f"{v:.4f}").to_markdown()
    return (
        f"# MovieLens 100k — top-{K} evaluation\n\n"
        f"Reproduce with `python scripts/benchmark.py` "
        f"(seed = {SEED}, 80/20 split).\n\n"
        f"{body}\n"
    )


def render_chart(results: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    results.plot.bar(ax=ax, edgecolor="white", linewidth=0.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_ylabel("Score")
    ax.set_title(f"MovieLens 100k — top-{K} evaluation (seed={SEED})")
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=144)
    plt.close(fig)


def main() -> None:
    print("Loading MovieLens 100k ...")
    ratings = load_movielens_100k()
    train, test = split_ratings(ratings, test_size=0.2, random_state=SEED)
    catalog = set(ratings["item_id"].unique())

    rows = {}
    for name, model in build_models().items():
        print(f"  evaluating {name} ...")
        rows[name] = evaluate(model, train, test, catalog)

    results = pd.DataFrame(rows).T
    OUT_DIR.mkdir(exist_ok=True)

    table_path = OUT_DIR / "results.md"
    chart_path = OUT_DIR / "results.png"
    table_path.write_text(render_table(results))
    render_chart(results, chart_path)

    print()
    print(results.to_string(float_format=lambda v: f"{v:.4f}"))
    print()
    print(f"wrote {table_path.relative_to(OUT_DIR.parent)}")
    print(f"wrote {chart_path.relative_to(OUT_DIR.parent)}")


if __name__ == "__main__":
    main()
