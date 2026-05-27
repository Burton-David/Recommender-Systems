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

    Raises
    ------
    ValueError
        If ``path`` does not contain a valid pickle stream.
    TypeError
        If the loaded object is not a :class:`Recommender`.
    """
    raw = Path(path).read_bytes()
    try:
        obj = pickle.loads(raw)
    except (pickle.UnpicklingError, EOFError) as exc:
        raise ValueError(f"{path} is not a valid recommender file") from exc
    if not isinstance(obj, Recommender):
        raise TypeError(f"{path} does not contain a Recommender")
    return obj
