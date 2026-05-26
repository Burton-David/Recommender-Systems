"""Ranking metrics for top-N recommendation evaluation.

All metrics take parallel sequences of predicted ranked lists and held-out
relevant item sets — one entry per user — and return a macro-averaged score
across users that have at least one held-out item. Users with no relevant
items are skipped from the mean so they don't drag every metric to zero.
"""

from collections.abc import Collection, Hashable, Sequence
from math import log2

__all__ = [
    "mean_average_precision",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
]


def _validate(
    predicted: Sequence[Sequence[Hashable]],
    actual: Sequence[Collection[Hashable]],
    k: int,
) -> None:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if len(predicted) != len(actual):
        raise ValueError(
            f"predicted and actual must have the same length, "
            f"got {len(predicted)} and {len(actual)}"
        )


def precision_at_k(
    predicted: Sequence[Sequence[Hashable]],
    actual: Sequence[Collection[Hashable]],
    k: int,
) -> float:
    """Macro-averaged precision@k across users.

    Parameters
    ----------
    predicted
        For each user, an ordered iterable of recommended item ids
        (most relevant first).
    actual
        For each user, the items considered relevant in the holdout.
    k
        Cutoff; only the first ``k`` recommendations per user are considered.

    Returns
    -------
    float
        Mean of ``hits / k`` across users with at least one relevant item;
        ``0.0`` if no such users exist.
    """
    _validate(predicted, actual, k)
    scores: list[float] = []
    for preds, truth in zip(predicted, actual, strict=True):
        if not truth:
            continue
        truth_set = set(truth)
        hits = sum(1 for item in list(preds)[:k] if item in truth_set)
        scores.append(hits / k)
    return sum(scores) / len(scores) if scores else 0.0


def recall_at_k(
    predicted: Sequence[Sequence[Hashable]],
    actual: Sequence[Collection[Hashable]],
    k: int,
) -> float:
    """Macro-averaged recall@k across users.

    Parameters
    ----------
    predicted
        Per-user ranked predictions.
    actual
        Per-user relevant items.
    k
        Cutoff applied to the predicted lists.

    Returns
    -------
    float
        Mean of ``hits / |relevant|`` across users with at least one relevant
        item; ``0.0`` if no such users exist.
    """
    _validate(predicted, actual, k)
    scores: list[float] = []
    for preds, truth in zip(predicted, actual, strict=True):
        if not truth:
            continue
        truth_set = set(truth)
        hits = sum(1 for item in list(preds)[:k] if item in truth_set)
        scores.append(hits / len(truth_set))
    return sum(scores) / len(scores) if scores else 0.0


def mean_average_precision(
    predicted: Sequence[Sequence[Hashable]],
    actual: Sequence[Collection[Hashable]],
    k: int,
) -> float:
    """Mean Average Precision at ``k``.

    For each user with at least one relevant item,
    ``AP@k = (1 / min(|relevant|, k)) * sum_{i=1..k} rel(i) * precision_at_i``,
    where ``rel(i)`` is 1 if the i-th recommendation is relevant. The
    ``min(|relevant|, k)`` denominator keeps the per-user score in ``[0, 1]``
    when ``|relevant|`` exceeds the cutoff.
    """
    _validate(predicted, actual, k)
    scores: list[float] = []
    for preds, truth in zip(predicted, actual, strict=True):
        if not truth:
            continue
        truth_set = set(truth)
        hits = 0
        precision_sum = 0.0
        for i, item in enumerate(list(preds)[:k], start=1):
            if item in truth_set:
                hits += 1
                precision_sum += hits / i
        denom = min(len(truth_set), k)
        scores.append(precision_sum / denom)
    return sum(scores) / len(scores) if scores else 0.0


def ndcg_at_k(
    predicted: Sequence[Sequence[Hashable]],
    actual: Sequence[Collection[Hashable]],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain at ``k`` with binary relevance.

    ``DCG = sum_{i=1..k} rel(i) / log2(i + 1)``; ``IDCG`` is the best
    possible DCG for the user (all relevant items ranked first). Per-user
    NDCG is ``DCG / IDCG`` (or 0 when no relevant items exist), averaged
    across users with relevant items.
    """
    _validate(predicted, actual, k)
    scores: list[float] = []
    for preds, truth in zip(predicted, actual, strict=True):
        if not truth:
            continue
        truth_set = set(truth)
        dcg = sum(
            1.0 / log2(i + 1)
            for i, item in enumerate(list(preds)[:k], start=1)
            if item in truth_set
        )
        ideal_hits = min(len(truth_set), k)
        idcg = sum(1.0 / log2(i + 1) for i in range(1, ideal_hits + 1))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return sum(scores) / len(scores) if scores else 0.0
