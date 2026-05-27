import math

import pytest

from recommender_systems.metrics import (
    catalog_coverage,
    intra_list_diversity,
    mean_average_precision,
    ndcg_at_k,
    novelty,
    precision_at_k,
    recall_at_k,
    serendipity_at_k,
)

ALL_METRICS = [precision_at_k, recall_at_k, mean_average_precision, ndcg_at_k]


def test_precision_at_k_basic():
    predicted = [["A", "B", "C", "D", "E"], ["X", "Y", "Z"]]
    actual = [{"A", "C", "F"}, {"A"}]
    # user 1: top-3 = [A,B,C], 2 hits of 3; user 2: 0 hits of 3
    assert precision_at_k(predicted, actual, k=3) == pytest.approx((2 / 3 + 0) / 2)


def test_recall_at_k_basic():
    predicted = [["A", "B", "C", "D", "E"], ["X", "Y", "Z"]]
    actual = [{"A", "C", "F"}, {"A"}]
    # user 1: 2 of 3 relevant captured; user 2: 0 of 1
    assert recall_at_k(predicted, actual, k=3) == pytest.approx((2 / 3 + 0) / 2)


def test_perfect_predictions_score_one():
    predicted = [["A", "B", "C"]]
    actual = [{"A", "B", "C"}]
    for metric in ALL_METRICS:
        assert metric(predicted, actual, k=3) == pytest.approx(1.0), metric.__name__


def test_no_overlap_scores_zero():
    predicted = [["X", "Y", "Z"]]
    actual = [{"A", "B"}]
    for metric in ALL_METRICS:
        assert metric(predicted, actual, k=3) == 0.0, metric.__name__


def test_map_worked_example():
    # predicted=[A, B, C, D, E], actual={A, C, F}, k=3
    # i=1: A relevant, hits=1, p=1
    # i=2: B not relevant
    # i=3: C relevant, hits=2, p=2/3
    # denom = min(3, 3) = 3 → AP = (1 + 2/3) / 3 = 5/9
    predicted = [["A", "B", "C", "D", "E"]]
    actual = [{"A", "C", "F"}]
    assert mean_average_precision(predicted, actual, k=3) == pytest.approx(5 / 9)


def test_ndcg_worked_example():
    # predicted=[A, X, B], actual={A, B}, k=3
    # DCG = 1/log2(2) + 0 + 1/log2(4)
    # IDCG = 1/log2(2) + 1/log2(3)
    predicted = [["A", "X", "B"]]
    actual = [{"A", "B"}]
    dcg = 1.0 + 1.0 / math.log2(4)
    idcg = 1.0 + 1.0 / math.log2(3)
    assert ndcg_at_k(predicted, actual, k=3) == pytest.approx(dcg / idcg)


def test_ndcg_is_one_for_perfect_ranking_under_partial_capture():
    # All relevant items are at the top, even if k > |relevant|.
    predicted = [["A", "B", "X", "Y"]]
    actual = [{"A", "B"}]
    assert ndcg_at_k(predicted, actual, k=4) == pytest.approx(1.0)


def test_users_without_relevant_items_are_skipped():
    predicted = [["A", "B"], ["C", "D"]]
    actual = [{"A"}, set()]
    # only user 1 counts toward the mean
    assert precision_at_k(predicted, actual, k=2) == pytest.approx(0.5)
    assert recall_at_k(predicted, actual, k=2) == pytest.approx(1.0)
    assert mean_average_precision(predicted, actual, k=2) == pytest.approx(1.0)


def test_empty_inputs_return_zero():
    for metric in ALL_METRICS:
        assert metric([], [], k=5) == 0.0, metric.__name__


def test_short_predicted_list_is_handled():
    # predicted shorter than k: missing positions count as non-relevant
    predicted = [["A"]]
    actual = [{"A", "B"}]
    assert precision_at_k(predicted, actual, k=5) == pytest.approx(1 / 5)
    assert recall_at_k(predicted, actual, k=5) == pytest.approx(1 / 2)


def test_works_with_integer_item_ids():
    predicted = [[10, 20, 30]]
    actual = [{20, 40}]
    assert precision_at_k(predicted, actual, k=3) == pytest.approx(1 / 3)
    assert recall_at_k(predicted, actual, k=3) == pytest.approx(1 / 2)


@pytest.mark.parametrize("metric", ALL_METRICS)
def test_invalid_k_raises(metric):
    with pytest.raises(ValueError, match="k must be positive"):
        metric([["A"]], [{"A"}], k=0)
    with pytest.raises(ValueError, match="k must be positive"):
        metric([["A"]], [{"A"}], k=-1)


@pytest.mark.parametrize("metric", ALL_METRICS)
def test_mismatched_lengths_raise(metric):
    with pytest.raises(ValueError, match="same length"):
        metric([["A"], ["B"]], [{"A"}], k=1)


# ---- beyond-accuracy metrics ----


def _toy_similarity(a, b):
    # sim("a","b")=0.5, sim("a","c")=0.0, sim("b","c")=0.5; identical pairs → 1.0
    table = {
        ("a", "b"): 0.5,
        ("a", "c"): 0.0,
        ("b", "c"): 0.5,
    }
    if a == b:
        return 1.0
    return table.get((a, b), table.get((b, a), 0.0))


def test_intra_list_diversity_worked_example():
    # top-3 = [a, b, c]; pair distances: (a,b)=0.5, (a,c)=1.0, (b,c)=0.5
    # mean dissimilarity = (0.5 + 1.0 + 0.5) / 3 = 2/3
    predicted = [["a", "b", "c"]]
    assert intra_list_diversity(predicted, _toy_similarity, k=3) == pytest.approx(2 / 3)


def test_intra_list_diversity_skips_singleton_lists():
    # only user 2 contributes (user 1's top-1 has no pairs to compare)
    predicted = [["a"], ["a", "b"]]
    # user 2: one pair (a,b), distance = 0.5
    assert intra_list_diversity(predicted, _toy_similarity, k=2) == pytest.approx(0.5)


def test_intra_list_diversity_identical_items_have_zero_diversity():
    predicted = [["a", "a"]]
    assert intra_list_diversity(predicted, _toy_similarity, k=2) == pytest.approx(0.0)


def test_novelty_worked_example():
    # popularity: a=1/2, b=1/4, c=1/8 → -log2: 1, 2, 3
    popularity = {"a": 0.5, "b": 0.25, "c": 0.125}
    predicted = [["a", "b", "c"]]
    assert novelty(predicted, popularity, k=3) == pytest.approx((1 + 2 + 3) / 3)


def test_novelty_skips_unknown_items():
    popularity = {"a": 0.5}
    predicted = [["a", "unknown"]]
    # only "a" contributes → mean over 1 item = -log2(0.5) = 1
    assert novelty(predicted, popularity, k=2) == pytest.approx(1.0)


def test_novelty_skips_users_whose_topk_is_all_unknown():
    popularity = {"a": 0.5}
    predicted = [["a"], ["x", "y"]]
    # only user 1 contributes
    assert novelty(predicted, popularity, k=2) == pytest.approx(1.0)


def test_catalog_coverage_worked_example():
    catalog = {"a", "b", "c", "d"}
    predicted = [["a", "b"], ["a", "c"]]
    # union recommended = {a, b, c}, catalog = 4 → 3/4
    assert catalog_coverage(predicted, catalog, k=2) == pytest.approx(0.75)


def test_catalog_coverage_respects_k_cutoff():
    catalog = {"a", "b", "c"}
    predicted = [["a", "b", "c"]]
    # k=1 → only "a" counts → 1/3
    assert catalog_coverage(predicted, catalog, k=1) == pytest.approx(1 / 3)


def test_catalog_coverage_empty_catalog_returns_zero():
    assert catalog_coverage([["a", "b"]], set(), k=5) == 0.0


def test_catalog_coverage_ignores_off_catalog_recommendations():
    catalog = {"a", "b"}
    predicted = [["a", "ghost"]]
    # "ghost" is dropped; only "a" hits the catalog → 1/2
    assert catalog_coverage(predicted, catalog, k=2) == pytest.approx(0.5)


def test_serendipity_worked_example():
    # truth = {A, B}; expected = {A, X}; predicted = [A, B, C], k=3
    # A: relevant but expected → not serendipitous
    # B: relevant and unexpected → serendipitous
    # C: not relevant → no
    # hits = 1 → score = 1/3
    predicted = [["A", "B", "C"]]
    actual = [{"A", "B"}]
    expected = [{"A", "X"}]
    assert serendipity_at_k(predicted, actual, expected, k=3) == pytest.approx(1 / 3)


def test_serendipity_empty_expected_is_just_relevance():
    predicted = [["A", "B", "C"]]
    actual = [{"A", "B"}]
    expected = [set()]
    # nothing is "expected", so every relevant hit is serendipitous → 2/3
    assert serendipity_at_k(predicted, actual, expected, k=3) == pytest.approx(2 / 3)


def test_serendipity_users_without_relevant_items_are_skipped():
    predicted = [["A"], ["B"]]
    actual = [{"A"}, set()]
    expected = [set(), set()]
    # only user 1 contributes → 1/1 = 1.0
    assert serendipity_at_k(predicted, actual, expected, k=1) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "metric, args",
    [
        (intra_list_diversity, ([["a"]], _toy_similarity)),
        (novelty, ([["a"]], {"a": 0.5})),
        (catalog_coverage, ([["a"]], {"a"})),
    ],
)
def test_beyond_accuracy_invalid_k_raises(metric, args):
    with pytest.raises(ValueError, match="k must be positive"):
        metric(*args, k=0)


def test_serendipity_invalid_k_raises():
    with pytest.raises(ValueError, match="k must be positive"):
        serendipity_at_k([["a"]], [{"a"}], [set()], k=0)


def test_serendipity_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="same length"):
        serendipity_at_k([["a"], ["b"]], [{"a"}], [set()], k=1)
