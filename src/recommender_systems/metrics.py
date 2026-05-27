"""Metrics for top-N recommendation evaluation.

Accuracy metrics (precision, recall, MAP, NDCG) take parallel sequences of
predicted ranked lists and held-out relevant item sets — one entry per user —
and return a macro-averaged score across users that have at least one held-out
item. Beyond-accuracy metrics (diversity, novelty, coverage, serendipity)
capture different aspects of recommendation quality and are documented inline.
Users with no relevant items are skipped from each mean so they don't drag the
score toward zero.
"""

from collections.abc import Callable, Collection, Hashable, Mapping, Sequence
from itertools import combinations
from math import log2

__all__ = [
    "catalog_coverage",
    "intra_list_diversity",
    "mean_average_precision",
    "ndcg_at_k",
    "novelty",
    "precision_at_k",
    "recall_at_k",
    "serendipity_at_k",
]


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")


def _validate(
    predicted: Sequence[Sequence[Hashable]],
    actual: Sequence[Collection[Hashable]],
    k: int,
) -> None:
    _validate_k(k)
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


def intra_list_diversity(
    predicted: Sequence[Sequence[Hashable]],
    similarity: Callable[[Hashable, Hashable], float],
    k: int,
) -> float:
    """Macro-averaged intra-list dissimilarity across users.

    For each user, dissimilarity averages ``1 - similarity(a, b)`` over all
    distinct pairs in the top-k recommendations. Users with fewer than two
    recommendations are skipped — diversity is undefined for singletons.

    Parameters
    ----------
    predicted
        Per-user ranked predictions.
    similarity
        Symmetric similarity in ``[0, 1]``. Only off-diagonal pairs are evaluated.
    k
        Cutoff for the top of each prediction list.
    """
    _validate_k(k)
    scores: list[float] = []
    for preds in predicted:
        top_k = list(preds)[:k]
        if len(top_k) < 2:
            continue
        pairs = list(combinations(top_k, 2))
        scores.append(sum(1.0 - similarity(a, b) for a, b in pairs) / len(pairs))
    return sum(scores) / len(scores) if scores else 0.0


def novelty(
    predicted: Sequence[Sequence[Hashable]],
    item_popularity: Mapping[Hashable, float],
    k: int,
) -> float:
    """Mean self-information of recommended items, averaged across users.

    Self-information of item ``i`` is ``-log2(p_i)``; popular items contribute
    little novelty, rare ones contribute more. Items absent from
    ``item_popularity`` are skipped within a user's list; users whose entire
    top-k is unknown are skipped from the mean.

    Parameters
    ----------
    predicted
        Per-user ranked predictions.
    item_popularity
        Mapping from item id to its popularity in ``(0, 1]`` (e.g., the
        fraction of users that have interacted with it).
    k
        Cutoff for the top of each prediction list.
    """
    _validate_k(k)
    scores: list[float] = []
    for preds in predicted:
        contribs = [
            -log2(item_popularity[item]) for item in list(preds)[:k] if item in item_popularity
        ]
        if not contribs:
            continue
        scores.append(sum(contribs) / len(contribs))
    return sum(scores) / len(scores) if scores else 0.0


def catalog_coverage(
    predicted: Sequence[Sequence[Hashable]],
    catalog: Collection[Hashable],
    k: int,
) -> float:
    """Fraction of the catalog that appears in at least one user's top-k.

    Items recommended outside ``catalog`` are ignored; recommending the same
    item to many users still only counts it once.

    Parameters
    ----------
    predicted
        Per-user ranked predictions.
    catalog
        Items the recommender could have recommended.
    k
        Cutoff applied to each user's prediction list.
    """
    _validate_k(k)
    if not catalog:
        return 0.0
    catalog_set = set(catalog)
    recommended: set[Hashable] = set()
    for preds in predicted:
        recommended.update(item for item in list(preds)[:k] if item in catalog_set)
    return len(recommended) / len(catalog_set)


def serendipity_at_k(
    predicted: Sequence[Sequence[Hashable]],
    actual: Sequence[Collection[Hashable]],
    expected: Sequence[Collection[Hashable]],
    k: int,
) -> float:
    """Macro-averaged share of top-k that are both relevant and unexpected.

    A recommendation counts as serendipitous when the user finds it relevant
    (``item in actual``) and a baseline recommender would *not* have surfaced
    it (``item not in expected``). The ``expected`` list per user is typically
    drawn from a popularity or otherwise trivial recommender. Users with no
    relevant items are skipped from the mean.

    Parameters
    ----------
    predicted
        Per-user ranked predictions.
    actual
        Per-user relevant items.
    expected
        Per-user baseline recommendations; items already inside this set are
        not serendipitous even if relevant.
    k
        Cutoff applied to each user's prediction list.
    """
    _validate_k(k)
    if not (len(predicted) == len(actual) == len(expected)):
        raise ValueError(
            f"predicted, actual, and expected must all have the same length, "
            f"got {len(predicted)}, {len(actual)}, {len(expected)}"
        )
    scores: list[float] = []
    for preds, truth, baseline in zip(predicted, actual, expected, strict=True):
        if not truth:
            continue
        truth_set = set(truth)
        baseline_set = set(baseline)
        hits = sum(1 for item in list(preds)[:k] if item in truth_set and item not in baseline_set)
        scores.append(hits / k)
    return sum(scores) / len(scores) if scores else 0.0
