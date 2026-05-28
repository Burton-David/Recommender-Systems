# Choosing an algorithm

Every recommender in the library implements the same `fit(ratings)` /
`recommend(user, n)` contract, so swapping one for another is one line. But
they're not interchangeable on the same problem — each has a setting where
it shines and one where it falls over. Here's the short version.

## At a glance

| Algorithm | What it needs | When it's the right call | When it isn't |
|-----------|---------------|--------------------------|---------------|
| `MostPopular` | Just ratings | Cold-start baseline; a reasonable default for new users | Personalization (it gives everyone the same list) |
| `MeanRating` | Ratings + an explicit-rating scale | When highly-rated long-tail items matter | Implicit feedback; low catalog coverage |
| `ItemKNN` | Ratings; sparse item-item cosine | Strong CF baseline; broad catalog coverage | Cold-start items (an item with no ratings is unreachable) |
| `UserKNN` | Ratings; per-user top-k neighbors via `NearestNeighbors` | Communities and tight clusters | When you specifically want a fully precomputed user-user matrix for downstream analysis |
| `SVD` | Ratings; `TruncatedSVD` over the sparse user-item matrix | Dense latent structure; smoothing | Implicit feedback where 0 is not negative |
| `BPR` | Implicit interactions only | Implicit feedback; learning what *order* items rank in | When you need closed-form solves rather than SGD — reach for ALS |
| `ALS` | Implicit interactions only | Same setting as BPR; converges in many fewer epochs | Very high `n_factors` where the per-side solves dominate |
| `ContentBased` | Per-item features | Cold-start items, niche-content surfacing | A pure-CF win on accuracy is on the table |
| `TwoTowerCF` | Implicit interactions; PyTorch | Embedding models with future side-information towers | Most cases — BPR/ALS are simpler and often as strong |
| `HybridRecommender` | Two or more fitted recommenders | Blending CF with content (cold-start) or model classes | One signal is dramatically stronger — weights matter |

## When pure collaborative filtering wins

If you have plenty of interactions per user and the catalog is reasonably
small/dense, neighborhood CF (`ItemKNN`) or matrix factorization (`SVD`,
`ALS`) is hard to beat on accuracy. The MovieLens benchmark is a canonical
example: ratings are dense, the catalog is bounded, and `ItemKNN` / `UserKNN`
/ `SVD` cluster within 1-2% of each other on precision and NDCG.

## When you need content

Three signs:

1. **Cold-start items.** A new item with zero ratings is invisible to CF;
   `ContentBased` can score it from features alone.
2. **Long-tail discovery.** CF tends to collapse onto popular items.
   Content surfaces niches that share features with what a user already
   liked.
3. **Explainability.** `ContentBased.recommend_with_reasons` returns a
   per-recommendation reason ("shared tags: fantasy, magic"), which CF
   can't.

`recommender_systems.books.build_tag_recommender` turns a goodbooks-style tag
table into a `ContentBased` via TF-IDF in one call. Compose it with a CF
recommender by hand through `HybridRecommender(...)` for a hybrid pipeline.

## When you need a learned latent space

When the data is large enough that `n_factors << min(n_users, n_items)` is a
useful compression, and you want a single dense user/item embedding you can
use elsewhere (clustering, search, downstream models), reach for `SVD`,
`ALS`, or `TwoTowerCF`. `SVD` is the simplest (truncated SVD on the rating
matrix); `ALS` converges in fewer epochs at the cost of per-epoch work;
`TwoTowerCF` extends naturally if you want to drop side information into
either tower.

## When you need implicit-feedback ranking

`BPR` and `ALS` both target the implicit setting (presence of an
interaction, not its rating). They optimize different objectives —
sigmoid-margin ranking loss (BPR) vs confidence-weighted regularized least
squares (ALS) — and converge differently. `ALS` gets close to its final
quality in 10-15 epochs; `BPR` needs more passes but is simpler to extend
with custom samplers.

## Composing

`HybridRecommender` fuses two or more fitted recommenders' top-N output via
weighted reciprocal-rank fusion. The hybrid never needs to peek inside its
components, so anything implementing the `Recommender` interface composes —
including another hybrid.

## Reasonable defaults to start

```python
# Strong CF baseline.
ItemKNN(k=20)

# Explicit-feedback latent factor.
SVD(n_factors=50, random_state=0)

# Implicit feedback, fast to fit.
ALS(n_factors=32, epochs=15, random_state=0)

# Content-only or content-aware hybrid.
ContentBased(item_features=tfidf_or_embeddings)
```

These match what `scripts/benchmark.py` and `scripts/benchmark_goodbooks.py`
ship with.
