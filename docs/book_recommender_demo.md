# Book recommender demo (goodbooks-10k)

End-to-end walkthrough of the book-recommender showcase: load goodbooks-10k,
shrink it to a tractable subset, build a hybrid of collaborative filtering and
tag-based content, fit, recommend, and explain.

!!! warning "Research data only"
    The [goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) dataset
    derives from Goodreads and is licensed for **benchmarking and research
    only** — it is not suitable for shipping in a commercial product. The
    [Open Library metadata client](https://github.com/Burton-David/Recommender-Systems/issues/48)
    is the planned product-path swap.

## Setup

```bash
pip install -e ".[benchmarks]"   # for the chart-generating helpers
```

## 1. Load the ratings, the tag table, and trim the corpus

```python
from recommender_systems import densest_subset, holdout_per_user
from recommender_systems.datasets import load_goodbooks_10k, load_goodbooks_tags

ratings = load_goodbooks_10k().rename(columns={"book_id": "item_id"})
tags = load_goodbooks_tags()

# Full corpus is ~53k users x 10k books — dense user-user similarity would need
# ~22 GB. Restrict to the most-active 2500 users and most-popular 3000 books so
# every algorithm fits in memory.
ratings = densest_subset(ratings, n_users=2500, n_items=3000)
```

## 2. Split, then fit a hybrid

```python
from recommender_systems.books import build_hybrid_book_recommender

train, test = holdout_per_user(ratings, test_size=0.2, random_state=20260527)

# Hybrid of ItemKNN (default collaborative) and a tag-based ContentBased,
# fused by HybridRecommender's weighted RRF.
recommender = build_hybrid_book_recommender(tags, max_features=200).fit(train)
```

## 3. Recommend and evaluate

```python
from recommender_systems.metrics import ndcg_at_k, precision_at_k

users = test["user_id"].unique()
predicted = [recommender.recommend(u, n=10) for u in users]
truth = test.groupby("user_id")["item_id"].agg(set)
actual = [truth.get(u, set()) for u in users]

print(f"precision@10 = {precision_at_k(predicted, actual, k=10):.3f}")
print(f"NDCG@10      = {ndcg_at_k(predicted, actual, k=10):.3f}")
```

## 4. Explain *why* a book was recommended

The hybrid's content half (a TF-IDF over book tags) carries human-readable
labels. Get them via the underlying `ContentBased`:

```python
content = recommender.recommenders[1]   # collab is index 0, content is index 1
user = int(users[0])

for item, reason in content.recommend_with_reasons(user, n=5):
    print(f"  {item:>6}  {reason}")
```

Sample output:

```
  47    science fiction, dystopia, post apocalyptic
  64    fantasy, magic, high fantasy
  101   thriller, mystery, suspense
  ...
```

The reason is the comma-separated list of features whose product
(`user_profile_weight x item_feature_weight`) contributes most to the
content-similarity score; see `ContentBased.explain` in the API reference for
the precise definition.

## Where the benchmark numbers come from

The [goodbooks-10k benchmark table](https://github.com/Burton-David/Recommender-Systems#benchmarks)
runs exactly this pipeline (plus a few baselines) via
`python -m scripts.benchmark_goodbooks`. On the 2500-user subsample,
`HybridBook` lands in the top tier alongside `ItemKNN` — within roughly 5% on
precision and catalog coverage, ~10% behind on NDCG and MAP — while the
content half pulls its weight on items both signals agree on and opens a
fallback path for cold-start books the CF half has never seen.
