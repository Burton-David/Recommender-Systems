"""Save and load fitted recommenders."""

from __future__ import annotations

import pickle
from pathlib import Path

from recommender_systems.base import Recommender

__all__ = ["load", "save"]


def save(model: Recommender, path: str | Path) -> None:
    """Persist a fitted recommender to ``path`` using pickle."""
    Path(path).write_bytes(pickle.dumps(model))


def load(path: str | Path) -> Recommender:
    """Load a recommender previously saved with :func:`save`.

    Only load files you trust: unpickling executes arbitrary code.
    """
    obj = pickle.loads(Path(path).read_bytes())
    if not isinstance(obj, Recommender):
        raise TypeError(f"{path} does not contain a Recommender")
    return obj
