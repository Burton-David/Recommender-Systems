# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Rust+PyO3 kernel (`crates/recsys-kernels/`) for BPR's inner SGD loop.
  `BPR.fit` now calls into the compiled extension and falls back to the
  pure-Python loop only when the extension isn't importable.
  See `docs/evolution/02-rust-kernel-bpr.md`.

### Changed

- Build backend moved from Hatchling to maturin so the same
  `pip install -e .` flow compiles the Rust extension. Building from
  source now requires a Rust toolchain; CONTRIBUTING.md has the setup.
- `recommender_systems.__version__` is now sourced from installed
  package metadata rather than hard-coded in `__init__.py`.

## [0.1.0]

Initial pre-release `main`. Snapshot of what will become `0.1.0`.

### Added

- Common `Recommender` interface (`fit` / `recommend`) and the private
  `_MatrixBackedRecommender` base that owns the unknown-user / seen-mask /
  top-n contract.
- Recommenders:
  - `MostPopular`, `MeanRating` baselines.
  - `UserKNN`, `ItemKNN` cosine-similarity neighborhood CF.
  - `SVD` truncated-SVD matrix factorization.
  - `BPR` Bayesian Personalized Ranking (pure-numpy SGD).
  - `ALS` Hu/Koren/Volinsky 2008 implicit-feedback alternating least squares.
  - `ContentBased` item-feature similarity with side-information convention.
  - `HybridRecommender` weighted reciprocal-rank fusion across any
    recommenders.
  - `TwoTowerCF` two-tower neural CF (requires the `[neural]` extra).
- Data utilities: `build_user_item_matrix`,
  `build_sparse_user_item_matrix`, `split_ratings`, `holdout_per_user`,
  `densest_subset`.
- Dataset loaders: `load_movielens_100k`, `load_goodbooks_10k`,
  `load_goodbooks_books`, `load_goodbooks_tags`.
- Evaluation metrics: `precision_at_k`, `recall_at_k`,
  `mean_average_precision`, `ndcg_at_k`, plus beyond-accuracy
  `intra_list_diversity`, `novelty`, `catalog_coverage`,
  `serendipity_at_k`.
- Text-feature builder `text_features` (TF-IDF / count / binary).
- Book-specific helpers `tag_text_per_book`, `build_tag_recommender`,
  `build_hybrid_book_recommender`.
- Explainability: `ContentBased.explain` and
  `ContentBased.recommend_with_reasons`.
- Persistence: `persistence.save` and `persistence.load`.
- `recsys` CLI (`recommend`, `evaluate`, `list-algos`).
- Reproducible benchmarks on MovieLens 100k and goodbooks-10k with
  committed tables and charts; one-command regeneration via
  `python -m scripts.benchmark` / `python -m scripts.benchmark_goodbooks`.
- Material for MkDocs documentation site published to GitHub Pages.
- CI matrix on Python 3.10 / 3.11 / 3.12 with Ruff, mypy, pytest, and
  Codecov upload.
- PEP 561 `py.typed` marker so downstream type checkers pick up our
  annotations.

### Roadmap

Items tracked for `0.2.0` and later — see `ROADMAP.md` and the issue
tracker.

- Scale: scipy-sparse-aware variants of `UserKNN`, `ItemKNN`, and `SVD`
  so the goodbooks-10k benchmark can run the full corpus.
- Product path: Open Library metadata client and a first-party
  reading-signal model.
- Release: PyPI publication.
