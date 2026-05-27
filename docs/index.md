# Recommender Systems

A collection of classic and modern recommender system algorithms with a clean,
unified API.

## Highlights

- **One interface.** Every algorithm implements `fit(ratings)` and
  `recommend(user, n)`, so they're interchangeable.
- **Five algorithms out of the box.** `MostPopular`, `MeanRating`, `UserKNN`,
  `ItemKNN`, `SVD` — more on the way.
- **Evaluation built-in.** `precision@k`, `recall@k`, `MAP`, `NDCG`, plus
  beyond-accuracy metrics (intra-list diversity, novelty, catalog coverage,
  serendipity).
- **Datasets.** `load_movielens_100k()` downloads and caches the standard
  benchmark; CI tests stay offline.
- **Typed, tested, ruff-clean.** Python 3.10+; pandas / numpy / scikit-learn
  under the hood.

## Install

```bash
pip install recommender-systems
```

For the optional word-embeddings recommender:

```bash
pip install 'recommender-systems[embeddings]'
```

See [Quickstart](quickstart.md) for a worked example, or
[API Reference](api.md) for the full surface.
