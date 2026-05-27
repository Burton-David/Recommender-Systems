"""Command-line interface for the recommender_systems library.

Installed as the ``recsys`` console script via the project's ``[project.scripts]``
entry point. Two subcommands so far::

    recsys recommend --algo item-knn --user 42 --n 10
    recsys evaluate  --algo svd

Both train on MovieLens 100k (downloaded and cached on first use).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from recommender_systems import Recommender, split_ratings
from recommender_systems.baselines import MeanRating, MostPopular
from recommender_systems.datasets import load_movielens_100k
from recommender_systems.metrics import (
    mean_average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from recommender_systems.neighborhood import ItemKNN, UserKNN
from recommender_systems.svd import SVD

ALGORITHMS: dict[str, Callable[..., Recommender]] = {
    "most-popular": MostPopular,
    "mean-rating": MeanRating,
    "item-knn": ItemKNN,
    "user-knn": UserKNN,
    "svd": SVD,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recsys",
        description="Train, recommend, and evaluate with recommender-systems.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for splits and stochastic models (default: 0)",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    rec = sub.add_parser("recommend", help="Recommend items for a user")
    rec.add_argument("--algo", choices=sorted(ALGORITHMS), required=True)
    rec.add_argument("--user", type=int, required=True, help="User id to recommend for")
    rec.add_argument("--n", type=int, default=10, help="How many items to return (default: 10)")

    ev = sub.add_parser("evaluate", help="Train/test-split and report top-k metrics")
    ev.add_argument("--algo", choices=sorted(ALGORITHMS), required=True)
    ev.add_argument("--k", type=int, default=10, help="Cutoff for top-k metrics (default: 10)")
    ev.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows held out for evaluation (default: 0.2)",
    )

    sub.add_parser("list-algos", help="Print the algorithms recsys knows about")

    return parser


def _instantiate(name: str, seed: int) -> Recommender:
    factory = ALGORITHMS[name]
    if name == "svd":
        return factory(random_state=seed)
    return factory()


def _cmd_recommend(args: argparse.Namespace) -> int:
    ratings = load_movielens_100k()
    model = _instantiate(args.algo, args.seed).fit(ratings)
    items = model.recommend(args.user, n=args.n)
    if not items:
        print(f"No recommendations for user {args.user}.", file=sys.stderr)
        return 1
    for rank, item in enumerate(items, start=1):
        print(f"{rank}\t{item}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    ratings = load_movielens_100k()
    train, test = split_ratings(ratings, test_size=args.test_size, random_state=args.seed)
    model = _instantiate(args.algo, args.seed).fit(train)
    truth = test.groupby("user_id")["item_id"].agg(set)
    users = sorted(truth.index)
    predicted = [model.recommend(u, n=args.k) for u in users]
    actual = [truth[u] for u in users]
    metrics = {
        f"precision@{args.k}": precision_at_k(predicted, actual, args.k),
        f"recall@{args.k}": recall_at_k(predicted, actual, args.k),
        f"MAP@{args.k}": mean_average_precision(predicted, actual, args.k),
        f"NDCG@{args.k}": ndcg_at_k(predicted, actual, args.k),
    }
    for name, value in metrics.items():
        print(f"{name:14s} {value:.4f}")
    return 0


def _cmd_list_algos(_args: argparse.Namespace) -> int:
    for name in sorted(ALGORITHMS):
        print(name)
    return 0


COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "recommend": _cmd_recommend,
    "evaluate": _cmd_evaluate,
    "list-algos": _cmd_list_algos,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
