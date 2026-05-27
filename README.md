# Recommender Systems Library

[![CI](https://github.com/Burton-David/Recommender-Systems/actions/workflows/ci.yml/badge.svg)](https://github.com/Burton-David/Recommender-Systems/actions/workflows/ci.yml)
[![Docs](https://github.com/Burton-David/Recommender-Systems/actions/workflows/docs.yml/badge.svg)](https://burton-david.github.io/Recommender-Systems/)
[![codecov](https://codecov.io/gh/Burton-David/Recommender-Systems/branch/main/graph/badge.svg)](https://codecov.io/gh/Burton-David/Recommender-Systems)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

A collection of classic and modern recommender system algorithms with a unified API:
every algorithm implements `fit(ratings)` and `recommend(user, n)`, so they're
interchangeable. Typed, tested, and benchmarked.

## Benchmarks

Top-10 evaluation on **MovieLens 100k** (80/20 seeded split). Reproduce with
`pip install -e ".[dev,benchmarks]" && python -m scripts.benchmark`.

![MovieLens 100k benchmark](benchmarks/results.png)

|             | precision@10 | recall@10 |  MAP@10 | NDCG@10 | coverage@10 |
|:------------|-------------:|----------:|--------:|--------:|------------:|
| MostPopular |       0.1863 |    0.1191 |  0.1104 |  0.2141 |      0.0315 |
| MeanRating  |       0.0490 |    0.0194 |  0.0140 |  0.0428 |      0.0161 |
| ItemKNN     |       0.3188 |    0.2010 |  0.2486 |  0.3786 |      0.2866 |
| UserKNN     |       0.3175 |    0.2123 |  0.2503 |  0.3881 |      0.2134 |
| SVD         |       0.3016 |    0.2134 |  0.2283 |  0.3675 |      0.2717 |

See [`benchmarks/results.md`](benchmarks/results.md) for the table regenerated from
the latest run.

Top-10 evaluation on **goodbooks-10k** (2500-user subsample, 80/20 seeded split).
Reproduce with `python -m scripts.benchmark_goodbooks`.

![goodbooks-10k benchmark](benchmarks/goodbooks_results.png)

|             | precision@10 | recall@10 |  MAP@10 | NDCG@10 | coverage@10 |
|:------------|-------------:|----------:|--------:|--------:|------------:|
| MostPopular |       0.0985 |    0.0434 |  0.0482 |  0.1080 |      0.0035 |
| MeanRating  |       0.0042 |    0.0019 |  0.0011 |  0.0040 |      0.0014 |
| ItemKNN     |       0.3256 |    0.1511 |  0.2314 |  0.3719 |      0.3413 |
| UserKNN     |       0.2414 |    0.1113 |  0.1552 |  0.2766 |      0.1286 |
| SVD         |       0.2714 |    0.1229 |  0.1840 |  0.3142 |      0.0739 |

See [`benchmarks/goodbooks_results.md`](benchmarks/goodbooks_results.md) for the
freshly-regenerated table.

## Install

```bash
git clone https://github.com/Burton-David/Recommender-Systems
cd Recommender-Systems
pip install -e .
```

Optional extras:

- `[embeddings]` — gensim for word-embedding features
- `[benchmarks]` — matplotlib + tabulate for `scripts/benchmark.py`
- `[docs]` — mkdocs-material for building the docs site
- `[dev]` — ruff, mypy, pytest, pytest-cov

## Quickstart

```python
from recommender_systems import split_ratings
from recommender_systems.datasets import load_movielens_100k
from recommender_systems.svd import SVD
from recommender_systems.metrics import ndcg_at_k, precision_at_k

ratings = load_movielens_100k()
train, test = split_ratings(ratings, test_size=0.2, random_state=0)

model = SVD(n_factors=50, random_state=0).fit(train)

users = test["user_id"].unique()
predicted = [model.recommend(u, n=10) for u in users]
truth = test.groupby("user_id")["item_id"].agg(set)
actual = [truth.get(u, set()) for u in users]

print(f"precision@10 = {precision_at_k(predicted, actual, k=10):.3f}")
print(f"NDCG@10      = {ndcg_at_k(predicted, actual, k=10):.3f}")
```

Swap `SVD` for `UserKNN`, `MostPopular`, etc. — the rest of the script is
unchanged. Full quickstart at
<https://burton-david.github.io/Recommender-Systems/quickstart/>.

There's also a CLI:

```bash
recsys recommend --algo item-knn --user 42 --n 10
recsys evaluate  --algo svd
```

## Algorithms

| Module                              | Class                | Notes                                                |
|-------------------------------------|----------------------|------------------------------------------------------|
| `recommender_systems.baselines`     | `MostPopular`        | Rank by interaction count                            |
|                                     | `MeanRating`         | Rank by mean rating with a min-ratings threshold     |
| `recommender_systems.neighborhood`  | `UserKNN`, `ItemKNN` | Cosine-similarity neighborhood CF                    |
| `recommender_systems.svd`           | `SVD`                | Truncated SVD on the user-item matrix                |
| `recommender_systems.content`       | `ContentBased`       | Item-feature similarity (TF-IDF, tags, embeddings)   |
| `recommender_systems.bpr`           | `BPR`                | Bayesian Personalized Ranking (numpy SGD)            |
| `recommender_systems.neural`        | `TwoTowerCF`         | Two-tower neural CF (PyTorch; requires `[neural]`)   |

`recommender_systems.features.text_features` builds TF-IDF / count / binary
item-by-term matrices from per-item text, ready to pass to `ContentBased`.

Evaluation metrics — `precision@k`, `recall@k`, `MAP@k`, `NDCG@k`, plus the
beyond-accuracy set (intra-list diversity, novelty, catalog coverage,
serendipity) — live in `recommender_systems.metrics`.

## Development

```bash
pip install -e ".[dev]"

ruff check src tests
ruff format --check src tests
mypy
pytest
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the quality bar and
[`ROADMAP.md`](ROADMAP.md) for the current phase plan.
