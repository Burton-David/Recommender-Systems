# Recommender Systems

A modern, fully-typed Python recommender library with a clean, unified API.

## Highlights

- **One interface.** Every algorithm implements `fit(ratings)` and
  `recommend(user, n)`, so they're interchangeable.
- **Ten algorithms out of the box.** `MostPopular`, `MeanRating`, `UserKNN`,
  `ItemKNN`, `SVD`, `BPR`, `ALS`, `ContentBased`, `TwoTowerCF`, and
  `HybridRecommender`. See [API Reference](api.md).
- **Compiled hot paths where they earn it.** BPR's SGD inner loop is a Rust+PyO3
  kernel (~51× faster than the pure-Python fallback); UserKNN, ItemKNN, and SVD
  run on `scipy.sparse` so they scale to goodbooks-10k at full size. The Python
  fallback paths are kept so the library works without the compiled extension.
- **Evaluation built-in.** `precision@k`, `recall@k`, `MAP`, `NDCG`, plus
  beyond-accuracy metrics (intra-list diversity, novelty, catalog coverage,
  serendipity).
- **Datasets.** `load_movielens_100k()` and `load_goodbooks_10k()` download and
  cache the standard benchmarks; CI tests stay offline.
- **Typed, tested, ruff-clean.** Python 3.10+; pandas / numpy / scikit-learn /
  scipy under the hood.

## Install

The library is not on PyPI yet. From source:

```bash
git clone https://github.com/Burton-David/Recommender-Systems
cd Recommender-Systems
pip install -e .
```

A Rust toolchain is required to compile the kernel — `brew install rust` or
[rustup](https://rustup.rs/) covers it. `pip install` then invokes `maturin`
to build the extension.

For optional features:

```bash
pip install -e '.[neural]'       # PyTorch for TwoTowerCF
pip install -e '.[embeddings]'   # gensim for word-embedding features
pip install -e '.[benchmarks]'   # matplotlib + tabulate for the benchmark scripts
```

See [Quickstart](quickstart.md) for a worked example, or
[API Reference](api.md) for the full surface.
