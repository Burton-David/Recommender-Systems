# Phase 3 — sparse-aware UserKNN / ItemKNN / SVD

**Status:** shipped.
**Tag:** `v0.3.0` (planned).
**Code change:** new `_SparseMatrixBackedRecommender` base; `UserKNN`,
`ItemKNN`, and `SVD` rewritten on top of it. Closes #77.

## Context

The MovieLens benchmark fits in memory comfortably (1.6k items, 940 users —
a 940×1682 dense matrix is 12 MB). The goodbooks-10k benchmark does not:
the dense user-item matrix is 53k × 10k = 530M cells, 4 GB at float64, and
the dense user-user similarity needed by the old `UserKNN` would be
53k × 53k = 22 GB. Until now `scripts/benchmark_goodbooks.py` had to
subsample to a 2,500-user slice for `UserKNN` to fit — which biased every
algorithm in the comparison, not just the one that needed it.

`recommender_systems.data.build_sparse_user_item_matrix` (shipped earlier)
returns a CSR matrix and `pd.Index` views for the rows and columns. It has
been sitting unused inside the package — Phase 3 cashes that in.

## Decision

Add a sibling base class `_SparseMatrixBackedRecommender` and migrate the
three algorithms whose memory profile actually cares —
`UserKNN`, `ItemKNN`, `SVD` — over to it. `ALS`, `BPR`, `ContentBased`,
`TwoTowerCF` stay on the existing dense `_MatrixBackedRecommender` because
they aren't memory-constrained at the scales the library targets and a
unified migration would multiply the diff for no benefit.

### Per-algorithm choices

| Algorithm | Approach |
|---|---|
| `ItemKNN` | `cosine_similarity(matrix.T, dense_output=False)` gives a CSR `(n_items, n_items)` similarity; `_sparse_top_k_per_row` zeroes the diagonal and keeps the top-`k` per row. Scoring a user is `sim @ user_row.T`. |
| `UserKNN` | `sklearn.neighbors.NearestNeighbors(metric="cosine")` finds the top-`k` neighbors per user without materializing the full `n_users × n_users` matrix. Per-user score is the similarity-weighted average of the neighbors' rating rows. |
| `SVD` | `TruncatedSVD` already accepts CSR natively; the dense pivot was the only thing keeping it from scaling. Drop it. |

## Options considered

| Option | Rejected because |
|---|---|
| Switch the *base* `_MatrixBackedRecommender` to sparse everywhere | Would require touching ALS, BPR, ContentBased, TwoTowerCF, all their tests, and `_PredictedScoreRecommender`. The algorithms that don't benefit shouldn't pay the migration cost; two parallel bases is a small price. |
| Compute a full `n_users × n_users` sparse similarity for UserKNN and trim per row | The intermediate is the problem — even after trim, computing the full pairwise distance is O(users² × items), 53,424² × 10,000 ≈ 28 trillion ops for goodbooks. `NearestNeighbors`' brute-force search with cosine metric is the same asymptotic but skips materializing the full matrix and parallelizes the search. |
| Use an approximate-NN library (FAISS, HNSWLib) for UserKNN | Phase 3's goal is to unlock the full corpus, not to make k-NN sub-linear. Approximate methods are a Phase-4+ candidate once the exact path is shown to be the right shape. |
| Add a `sparse=True` flag on the dense classes | Hides the actual decision behind a parameter. The dense and sparse paths have different internal state (CSR vs DataFrame) and different memory characteristics; users picking these algorithms benefit from the choice being explicit at the class level. |

## What's in the change

- `src/recommender_systems/base.py` — new `_SparseMatrixBackedRecommender`
  with `_matrix: csr_matrix`, `_users: Index`, `_items: Index`, and a
  sparse-aware `recommend` that masks seen items off the user's nonzero
  columns.
- `src/recommender_systems/neighborhood.py` — fully rewritten. `ItemKNN`
  uses sparse cosine; `UserKNN` uses `NearestNeighbors`. Module-private
  `_sparse_top_k_per_row` helper.
- `src/recommender_systems/svd.py` — uses the sparse builder and the
  CSR-native `TruncatedSVD`.
- Existing `tests/test_neighborhood.py` and `tests/test_svd.py` pass
  unchanged — the public contract is preserved.

## Results

On MovieLens 100k (same `tests/benchmarks/test_fit_speed.py` suite), the
sparse rewrite is also *faster* on small data, because the sparse cosine
and the smaller `NearestNeighbors` workload beat the original dense path:

| Algorithm | Phase 1.1 (dense) | Phase 3 (sparse) |
|-----------|------------------:|-----------------:|
| SVD       | 46 ms             | **29 ms**        |
| UserKNN   | 74 ms             | **33 ms**        |
| ItemKNN   | 141 ms            | **85 ms**        |

The headline result is qualitative, not relative: `UserKNN`, `ItemKNN`,
and `SVD` can now run the full goodbooks-10k corpus. The 2,500-user
subsample in `scripts/benchmark_goodbooks.py` is still there only because
`ContentBased` / `HybridBook` retain dense feature matrices; the three
sparse-aware algorithms no longer need it. A follow-up could add a
`--full` mode that runs only the sparse-aware algorithms over the entire
corpus, but that's a separate change.

## What's not in scope

- `ContentBased`, `HybridBook`. Their dense user-profile and feature
  matrices would need a separate sparse pass; out of this phase's scope.
- `ALS`. Phase 1.1 measurements showed it's already fast enough; the
  dense per-row solve is well-conditioned at the scales we target. It
  earns a rewrite only if a future profile puts it on the critical path.
- Approximate nearest neighbors. Exact brute-force search at
  goodbooks-10k scale is tractable; adding an ANN dependency before the
  exact path is shown to be too slow would be a premature optimization.
- The persistence format. CSR matrices and `pd.Index` are both picklable,
  so the existing `save`/`load` flow continues to work — but pickles
  written *before* this change won't load on the new code because the
  internal layout shifted. Pre-1.0; documented in CHANGELOG.
