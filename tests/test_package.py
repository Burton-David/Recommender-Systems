import recommender_systems


def test_version_is_exposed():
    assert isinstance(recommender_systems.__version__, str)
    assert recommender_systems.__version__
