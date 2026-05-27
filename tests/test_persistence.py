import pickle

import numpy as np
import pandas as pd
import pytest

from recommender_systems import (
    BPR,
    SVD,
    ContentBased,
    HybridRecommender,
    ItemKNN,
    MeanRating,
    MostPopular,
    UserKNN,
)
from recommender_systems.persistence import load, save


def _ratings():
    rng = np.random.default_rng(0)
    rows = []
    for user in range(10):
        cluster = user // 5  # two clusters
        items = list(range(0, 5)) if cluster == 0 else list(range(5, 10))
        for item in rng.choice(items, size=3, replace=False):
            rows.append((user, int(item)))
    return pd.DataFrame(rows, columns=["user_id", "item_id"]).assign(rating=1)


def _content_features():
    return pd.DataFrame(
        {"action": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0], "drama": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]},
        index=range(10),
    )


def _build(cls):
    if cls is MeanRating:
        return cls(min_ratings=1)
    if cls is SVD:
        return cls(n_factors=2, random_state=0)
    if cls is BPR:
        return cls(n_factors=2, epochs=2, random_state=0)
    if cls is ContentBased:
        return cls(item_features=_content_features())
    if cls is HybridRecommender:
        return cls([MostPopular(), MeanRating(min_ratings=1)])
    return cls()


@pytest.mark.parametrize(
    "cls",
    [
        MostPopular,
        MeanRating,
        ItemKNN,
        UserKNN,
        SVD,
        BPR,
        ContentBased,
        HybridRecommender,
    ],
)
def test_round_trip_preserves_recommendations_for_every_recommender(cls, tmp_path):
    model = _build(cls).fit(_ratings())
    path = tmp_path / "model.pkl"

    save(model, path)
    loaded = load(path)

    assert loaded.recommend(0, n=5) == model.recommend(0, n=5), cls.__name__


def test_load_rejects_non_recommender(tmp_path):
    path = tmp_path / "bad.pkl"
    path.write_bytes(pickle.dumps({"not": "a model"}))

    with pytest.raises(TypeError, match="Recommender"):
        load(path)
