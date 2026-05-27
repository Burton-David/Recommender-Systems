"""Dump a trained SVD model to JSON for the serving experiment.

Both the FastAPI and Go services read the same JSON file so the only
difference between them is the runtime — not the model, not the data,
not the loading code's quirks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from recommender_systems.datasets import load_movielens_100k
from recommender_systems.svd import SVD


def main() -> int:
    parser = argparse.ArgumentParser(prog="export_model")
    parser.add_argument("--algo", choices=["svd"], default="svd")
    parser.add_argument("--n-factors", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    ratings = load_movielens_100k()
    model = SVD(n_factors=args.n_factors, random_state=args.seed).fit(ratings)

    payload = {
        "algo": args.algo,
        "n_factors": args.n_factors,
        "user_ids": [int(u) for u in model._users.tolist()],
        "item_ids": [int(i) for i in model._items.tolist()],
        # user_factors: (n_users, n_factors); item_factors: (n_factors, n_items).
        # We transpose item_factors at export time so both servers do a
        # single matmul against (n_items, n_factors)-shaped tensors.
        "user_factors": model._user_factors.astype(np.float32).tolist(),
        "item_factors_T": model._item_factors.T.astype(np.float32).tolist(),
    }

    args.out.write_text(json.dumps(payload))
    n_user, n_item = len(payload["user_ids"]), len(payload["item_ids"])
    print(f"wrote {args.out}  ({n_user} users, {n_item} items, k={args.n_factors})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
