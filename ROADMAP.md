# Roadmap

A modern, well-tested recommender library, with a book-recommender vertical built
on top of it.

## Shipped

- **Packaging & tooling:** `src/` layout, `pyproject.toml`, Ruff lint/format, mypy,
  pytest, pre-commit, CI across Python 3.10–3.12, and a published docs site.
- **Unified API:** a `Recommender` interface (`fit` / `recommend`) so every algorithm
  is interchangeable, two matrix-backed bases (dense / sparse), and shared data
  utilities (`build_user_item_matrix`, `build_sparse_user_item_matrix`,
  `split_ratings`, `holdout_per_user`, `densest_subset`).
- **Algorithms:** most-popular & mean-rating baselines, user/item k-NN, SVD matrix
  factorization, implicit-feedback BPR, ALS, content-based (TF-IDF / count / binary
  features), two-tower neural CF, and a reciprocal-rank-fusion hybrid.
- **Evaluation:** precision@k, recall@k, MAP, NDCG, plus beyond-accuracy metrics
  (diversity, novelty, coverage, serendipity).
- **Reproducible benchmarks** on MovieLens 100k and goodbooks-10k (committed tables
  + charts), a `recsys` CLI, and model persistence.
- **Book recommender pipeline** (epic #50): goodbooks-10k loaders, tag-based content
  recommendation, hybrid collaborative+content, explainable recommendations, and a
  worked end-to-end demo.
- **Rust+PyO3 kernel for BPR's SGD inner loop** (~51× faster fit on MovieLens 100k).
  See `docs/evolution/02-rust-kernel-bpr.md`.
- **Sparse-aware UserKNN / ItemKNN / SVD** so the three algorithms run goodbooks-10k
  at full scale (closes #77). See `docs/evolution/03-sparse-recommenders.md`.
- **Engineering ADRs** in `docs/evolution/`, including the
  [Phase 4 refusal](docs/evolution/04-neural-stays-pytorch.md) to rewrite
  `TwoTowerCF` and the
  [Phase 5 serving experiment + postmortem](docs/evolution/05-go-serving-postmortem.md).

## Next

- **Release:** version bump, `cibuildwheel` workflow for Linux/macOS/Windows
  pre-built wheels, trusted-publishing setup, first PyPI upload.
- **ContentBased / HybridBook on sparse features** so the goodbooks benchmark can
  drop the 2,500-user subsample entirely.
- **Product path (deferred):** an Open Library metadata client (#48) and a
  first-party reading-signal model (#49) for the e-reader — commercial-safe, no
  scraped data.
