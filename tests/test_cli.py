import pandas as pd
import pytest

from recommender_systems import cli


@pytest.fixture
def fake_ratings(monkeypatch):
    """Replace the MovieLens download with a tiny synthetic ratings frame."""

    def _ratings(*_args, **_kwargs):
        return pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3, 3, 3],
                "item_id": ["a", "b", "a", "c", "a", "b", "c"],
                "rating": [5, 4, 5, 3, 5, 2, 4],
            }
        )

    monkeypatch.setattr(cli, "load_movielens_100k", _ratings)
    return _ratings


def test_list_algos(capsys):
    code = cli.main(["list-algos"])
    out = capsys.readouterr().out
    assert code == 0
    assert "most-popular" in out
    assert "svd" in out


def test_recommend_prints_ranked_items(capsys, fake_ratings):
    code = cli.main(["recommend", "--algo", "most-popular", "--user", "1", "--n", "1"])
    out = capsys.readouterr().out
    assert code == 0
    assert out.strip().startswith("1\t")
    assert len(out.strip().splitlines()) == 1


def test_recommend_unknown_user_exits_nonzero(capsys, fake_ratings):
    # item-knn returns [] for users absent from the trained matrix.
    code = cli.main(["recommend", "--algo", "item-knn", "--user", "999"])
    err = capsys.readouterr().err
    assert code == 1
    assert "999" in err


def test_evaluate_prints_metric_lines(capsys, fake_ratings):
    code = cli.main(["evaluate", "--algo", "most-popular", "--k", "2", "--test-size", "0.3"])
    out = capsys.readouterr().out
    assert code == 0
    names = [line.split()[0] for line in out.strip().splitlines()]
    assert names == ["precision@2", "recall@2", "MAP@2", "NDCG@2"]


def test_invalid_algo_exits_via_argparse():
    with pytest.raises(SystemExit):
        cli.main(["recommend", "--algo", "nonexistent", "--user", "1"])


def test_missing_required_subcommand_exits():
    with pytest.raises(SystemExit):
        cli.main([])


def test_seed_propagates_to_svd(monkeypatch, fake_ratings):
    seen = {}

    class _Recorder:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def fit(self, _ratings):
            return self

        def recommend(self, *_args, **_kwargs):
            return ["a"]

    monkeypatch.setitem(cli.ALGORITHMS, "svd", _Recorder)
    code = cli.main(["--seed", "42", "recommend", "--algo", "svd", "--user", "1", "--n", "1"])
    assert code == 0
    assert seen == {"random_state": 42}


def test_bpr_is_registered_and_seeded():
    from recommender_systems.bpr import BPR

    assert "bpr" in cli.ALGORITHMS
    model = cli._instantiate("bpr", seed=7)
    assert isinstance(model, BPR)
    assert model.random_state == 7
