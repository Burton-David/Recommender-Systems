# Beyond accuracy

A top-N recommender that nails `precision@k` and `NDCG@k` can still be a bad
product: it recommends the same blockbusters to everyone, never surfaces the
long tail, and shows the user nothing they couldn't have found themselves.
`recommender_systems.metrics` ships four beyond-accuracy metrics so the
benchmark catches that.

All four take the same per-user shape as the ranking metrics — parallel
sequences of predicted ranked lists and (for serendipity) held-out relevant
items — and return a macro-averaged float across users.

## Intra-list diversity

How different the items inside one user's top-N are from each other.
Macro-averages `1 - similarity(item_a, item_b)` over distinct pairs in the
top-k. A list of ten near-identical items has diversity near zero; a list of
ten unrelated items has diversity near one.

```python
from recommender_systems.metrics import intra_list_diversity

def jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / max(1, len(sa | sb))

predicted = [["sci-fi space", "sci-fi space alien", "regency romance"]]
ild = intra_list_diversity(predicted, jaccard, k=3)
# Two sci-fi items overlap heavily; the romance pulls diversity up.
```

## Novelty

Mean self-information of the recommended items: how much the recommender
surfaces things a user couldn't have stumbled on by themselves. Items that
everyone interacts with carry little novelty; long-tail items carry more.

```python
from recommender_systems.metrics import novelty

# popularity is fraction of users who interacted with each item.
item_popularity = {"a": 0.5, "b": 0.25, "c": 0.05}
predicted = [["a", "b", "c"]]
n = novelty(predicted, item_popularity, k=3)
# c (rarely seen) contributes much more than a (everyone has seen it).
```

## Catalog coverage

The simplest of the four — fraction of the catalog any user ever sees in
their top-k. A recommender that fixates on a handful of popular items has
coverage near zero; one that exposes the long tail has high coverage.

```python
from recommender_systems.metrics import catalog_coverage

catalog = {"a", "b", "c", "d", "e"}
predicted = [["a", "b"], ["a", "c"], ["a", "d"]]
coverage = catalog_coverage(predicted, catalog, k=2)
# a, b, c, d appear in some user's top-2 — coverage 4/5 = 0.8.
```

The MovieLens and goodbooks benchmarks both report `coverage@10`. The most
striking observation: `MeanRating` with `min_ratings=5` covers about 1.6% of
the catalog; `ItemKNN` covers ~29%. Accuracy parity doesn't say anything
about reach.

## Serendipity

How often the top-N contains items a user finds **both** relevant **and**
unexpected — items they wouldn't have gotten from a popularity baseline.
Captures the difference between "accurate and obvious" and "accurate and
surprising".

```python
from recommender_systems.baselines import MostPopular
from recommender_systems.metrics import serendipity_at_k

# Per user: the top-N a trivial baseline would have given them.
baseline = MostPopular().fit(train)
expected = [set(baseline.recommend(u, n=10)) for u in users]

predicted = [model.recommend(u, n=10) for u in users]
actual = [truth.get(u, set()) for u in users]
ser = serendipity_at_k(predicted, actual, expected, k=10)
```

A `MostPopular` recommender has zero serendipity by construction — its
recommendations *are* the baseline.

## Two recommenders with the same `precision@10` are not the same

The MovieLens benchmark shows `UserKNN` and `ItemKNN` essentially tied on
accuracy (precision@10 of 0.3199 vs 0.3240) but with very different catalog
coverage (0.2170 vs 0.2919). The accuracy metrics say both retrieve correct
items; coverage tells you `ItemKNN` does so across a broader slice of the
library. Whether one is better depends on the product question, not the
precision score.
