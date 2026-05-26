# Roadmap

Turn this collection of scripts into a polished, modern, well-tested recommender systems
library worth showcasing.

## Phase 0 — Foundation (in progress)

Packaging (`pyproject.toml`, `src/` layout), Ruff lint/format, pytest, CI, and contributor
docs. No behavior change yet.

## Phase 1 — Migrate the legacy modules

Move each existing script into `recommender_systems/`, with a clean signature, type hints,
docstrings, input validation, and tests. Remove import-time side effects. One module per PR.

## Phase 2 — A unified API

Introduce a common interface (e.g. a `Recommender` base with `fit` / `recommend`) so every
algorithm is interchangeable, plus shared utilities for building user–item matrices and
train/test splits.

## Phase 3 — More algorithms and evaluation

- Baselines: most-popular, mean-rating.
- Neighborhood: user/item kNN collaborative filtering.
- Matrix factorization: SVD, ALS, and an implicit-feedback model (BPR).
- Content-based: a unified TF-IDF / embeddings recommender.
- Hybrid: combine collaborative and content signals.
- Evaluation: precision@k, recall@k, MAP, NDCG, coverage, and a `MovieLens` loader.

## Phase 4 — Documentation and examples

Worked examples, benchmarks across algorithms, and a published docs site.

## Known issues (carried over from the original code)

- `model_based_collaborative_filter.py` executes at import time instead of inside a guard.
- `get_recommended_items` has an inconsistent signature across modules.
- The README has duplicated and mismatched algorithm descriptions.
- No input validation and no tests on any existing module.
