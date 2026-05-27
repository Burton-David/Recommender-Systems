import pickle

import pandas as pd
import pytest

from recommender_systems.baselines import MostPopular
from recommender_systems.persistence import load, save


def test_round_trip_preserves_recommendations(tmp_path):
    ratings = pd.DataFrame(
        {"user_id": [1, 1, 2, 2], "item_id": ["a", "b", "a", "c"], "rating": [5, 4, 3, 2]}
    )
    model = MostPopular().fit(ratings)
    path = tmp_path / "model.pkl"

    save(model, path)
    loaded = load(path)

    assert loaded.recommend(2, n=5) == model.recommend(2, n=5)


def test_load_rejects_non_recommender(tmp_path):
    path = tmp_path / "bad.pkl"
    path.write_bytes(pickle.dumps({"not": "a model"}))

    with pytest.raises(TypeError, match="Recommender"):
        load(path)
