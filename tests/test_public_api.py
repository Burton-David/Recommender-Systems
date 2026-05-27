import recommender_systems
from recommender_systems import Recommender

EXPECTED_RECOMMENDERS = [
    "ALS",
    "BPR",
    "ContentBased",
    "HybridRecommender",
    "ItemKNN",
    "MeanRating",
    "MostPopular",
    "SVD",
    "UserKNN",
]


def test_recommenders_are_importable_from_the_package_root():
    for name in EXPECTED_RECOMMENDERS:
        obj = getattr(recommender_systems, name)
        assert issubclass(obj, Recommender), name
        assert name in recommender_systems.__all__


def test_core_utilities_are_exported():
    for name in ("build_user_item_matrix", "split_ratings"):
        assert callable(getattr(recommender_systems, name))
        assert name in recommender_systems.__all__
