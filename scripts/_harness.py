"""Shared evaluation harness for the benchmark scripts.

`run_benchmark` evaluates a dict of recommenders against a single ratings frame and
writes a markdown table + chart to ``out_dir``. Individual benchmark scripts under
``scripts/`` are thin entry points that supply the dataset and the model dict.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from recommender_systems import Recommender, split_ratings
from recommender_systems.metrics import (
    catalog_coverage,
    mean_average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def _evaluate(
    model: Recommender,
    train: pd.DataFrame,
    test: pd.DataFrame,
    catalog: set,
    *,
    k: int,
) -> dict[str, float]:
    model.fit(train)
    truth = test.groupby("user_id")["item_id"].agg(set)
    users = sorted(truth.index)
    predicted = [model.recommend(u, n=k) for u in users]
    actual = [truth[u] for u in users]
    return {
        f"precision@{k}": precision_at_k(predicted, actual, k),
        f"recall@{k}": recall_at_k(predicted, actual, k),
        f"MAP@{k}": mean_average_precision(predicted, actual, k),
        f"NDCG@{k}": ndcg_at_k(predicted, actual, k),
        f"coverage@{k}": catalog_coverage(predicted, catalog, k),
    }


def _render_chart(name: str, results: pd.DataFrame, k: int, seed: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    results.plot.bar(ax=ax, edgecolor="white", linewidth=0.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_ylabel("Score")
    ax.set_title(f"{name} — top-{k} evaluation (seed={seed})")
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=144)
    plt.close(fig)


def run_benchmark(
    name: str,
    ratings: pd.DataFrame,
    models: dict[str, Recommender],
    *,
    out_dir: Path,
    file_stem: str,
    script: str,
    k: int = 10,
    seed: int = 0,
    test_size: float = 0.2,
    notes: str = "",
) -> pd.DataFrame:
    """Run every model in ``models`` on a fresh split of ``ratings`` and persist outputs.

    Parameters
    ----------
    name
        Display name used in the table heading and chart title (e.g. ``"MovieLens 100k"``).
    ratings
        Long-format interactions with at least ``user_id``, ``item_id``, ``rating`` columns.
    models
        Mapping from display name to a ready-to-fit recommender.
    out_dir
        Directory to write ``{file_stem}.md`` and ``{file_stem}.png`` into.
    file_stem
        Stem for the output files (e.g. ``"results"`` or ``"goodbooks_results"``).
    script
        Path the user runs to reproduce the table; mentioned in the markdown header.
    k, seed, test_size
        Standard knobs.
    notes
        Optional extra paragraph appended to the markdown header — useful for
        dataset-specific caveats (e.g. subsampling).
    """
    train, test = split_ratings(ratings, test_size=test_size, random_state=seed)
    catalog = set(ratings["item_id"].unique())

    rows: dict[str, dict[str, float]] = {}
    for model_name, model in models.items():
        print(f"  evaluating {model_name} ...")
        rows[model_name] = _evaluate(model, train, test, catalog, k=k)
    results = pd.DataFrame(rows).T

    out_dir.mkdir(exist_ok=True)
    table_path = out_dir / f"{file_stem}.md"
    chart_path = out_dir / f"{file_stem}.png"

    body = results.to_markdown(floatfmt=".4f")
    extra = f"\n{notes.strip()}\n" if notes else ""
    table_path.write_text(
        f"# {name} — top-{k} evaluation\n\n"
        f"Reproduce with `{script}` (seed = {seed}, "
        f"{int((1 - test_size) * 100)}/{int(test_size * 100)} split).{extra}\n\n"
        f"{body}\n"
    )
    _render_chart(name, results, k, seed, chart_path)

    print()
    print(results.to_string(float_format=lambda v: f"{v:.4f}"))
    print()
    print(f"wrote {table_path}")
    print(f"wrote {chart_path}")
    return results
