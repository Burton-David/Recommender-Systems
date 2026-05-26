import math

import pytest

from recommender_systems.metrics import (
    mean_average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
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
