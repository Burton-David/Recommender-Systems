# Roadmap

A polished, modern, well-tested recommender systems library — and a real book recommender
built on top of it.

## Shipped

- **Packaging & tooling:** `src/` layout, `pyproject.toml`, Ruff lint/format, mypy, pytest,
  CI across Python 3.10–3.12, and a published docs site.
- **Unified API:** a `Recommender` interface (`fit` / `recommend`) so every algorithm is
  interchangeable, a `MatrixBackedRecommender` base, and shared data utilities
  (`build_user_item_matrix`, `split_ratings`, `holdout_per_user`, `densest_subset`).
- **Algorithms:** most-popular & mean-rating baselines, user/item k-NN, SVD matrix
  factorization, implicit-feedback BPR, ALS, content-based (TF-IDF / count / binary
  features), and a reciprocal-rank-fusion hybrid.
- **Evaluation:** precision@k, recall@k, MAP, NDCG, plus beyond-accuracy metrics
  (diversity, novelty, coverage, serendipity).
- **Reproducible benchmarks** on MovieLens 100k and goodbooks-10k (committed tables +
  charts), a `recsys` CLI, and model persistence.

## In progress — the book recommender showcase (epic #50)

A book recommender that powers a real e-reader app: goodbooks-10k benchmark, tag-based
content recommendation, two-tower neural CF, hybrid collaborative+content, explainable
recommendations, and a worked demo.

## Next

- **Scale:** scipy-sparse user-item matrices so the neighborhood and matrix-factorization
  models run on the full goodbooks-10k corpus, not just a subsample (#77).
- **Product path (deferred):** an Open Library metadata client (#48) and a first-party
  reading-signal model (#49) for the e-reader — commercial-safe, no scraped data.
- **Release:** publish to PyPI.
