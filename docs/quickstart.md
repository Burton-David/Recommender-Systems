# Quickstart

A complete fit → recommend → evaluate flow on MovieLens 100k.

```python
from recommender_systems import split_ratings
from recommender_systems.datasets import load_movielens_100k
from recommender_systems.svd import SVD
from recommender_systems.metrics import ndcg_at_k, precision_at_k

# 1. Load and split.
ratings = load_movielens_100k()
train, test = split_ratings(ratings, test_size=0.2, random_state=0)

# 2. Fit.
model = SVD(n_factors=50, random_state=0).fit(train)

# 3. Recommend for the held-out users.
test_users = test["user_id"].unique()
predicted = [model.recommend(u, n=10) for u in test_users]

# 4. Evaluate against the holdout.
truth = test.groupby("user_id")["item_id"].agg(set)
actual = [truth.get(u, set()) for u in test_users]

print(f"precision@10 = {precision_at_k(predicted, actual, k=10):.3f}")
print(f"NDCG@10      = {ndcg_at_k(predicted, actual, k=10):.3f}")
```

Swap `SVD(...)` for `UserKNN()`, `ItemKNN()`, `MostPopular()`, or
`MeanRating()` — every recommender exposes the same `fit` / `recommend`
contract, so the rest of the script is unchanged.

## Beyond accuracy

The same `predicted` and `actual` lists feed the beyond-accuracy metrics:

```python
from recommender_systems.metrics import catalog_coverage, novelty

catalog = ratings["item_id"].unique()
counts = ratings["item_id"].value_counts(normalize=True).to_dict()

print(f"coverage@10 = {catalog_coverage(predicted, catalog, k=10):.3f}")
print(f"novelty@10  = {novelty(predicted, counts, k=10):.2f}")
```
