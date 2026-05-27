# Phase 1.1 baseline — where the time actually goes

Phase 1.1 of the [evolution arc](../docs/evolution/) is "measure before
cutting." No code is rewritten in this phase. The numbers below are the
ground truth that justifies — or refutes — every optimization the later
phases propose.

## Environment

| | |
|---|---|
| Python | 3.14.3 |
| Platform | macOS 15.7.1, arm64 (Apple Silicon) |
| numpy | 2.4.6 |
| pandas | 3.0.3 |
| scikit-learn | 1.8.0 |
| scipy | 1.17.1 |

Hardware-sensitive numbers. Re-run on your machine before drawing conclusions
about absolute timings; the *ratios* are what carry across hardware.

## Reproducing

```bash
pip install -e ".[dev]"
pytest tests/benchmarks/ -m benchmark --benchmark-only \
    --benchmark-columns=mean,stddev,rounds --benchmark-warmup=on
```

Benchmarks are skipped by default (`-m 'not benchmark'` in `pyproject.toml`)
and excluded from CI — wall-clock timings are too noisy on shared runners.

## Fit time, MovieLens 100k

100,000 ratings, 943 users, 1,682 items. Iterative algorithms use 5 epochs
and `n_factors=16` so the suite runs in under a minute.

| Algorithm     | Mean fit time | vs. MostPopular |
|---------------|---------------|-----------------|
| MostPopular   | 16.1 ms       | 1.0×            |
| MeanRating    | 16.5 ms       | 1.0×            |
| SVD (k=20)    | 46.1 ms       | 2.9×            |
| UserKNN (k=20)| 74.0 ms       | 4.6×            |
| ItemKNN (k=20)| 140.9 ms      | 8.8×            |
| ALS (16/5)    | 473.2 ms      | 29×             |
| **BPR (16/5)**| **4,503.4 ms**| **280×**        |

**BPR dominates by an order of magnitude.** Everything else fits in under
half a second; BPR alone takes 4.5 seconds for the same dataset with the
same epoch budget.

## Where BPR spends its time

`cProfile`, sorted by cumulative time:

```
         507806 function calls in 4.687 seconds

   ncalls  tottime  percall  cumtime  percall function
        1    0.674    0.674    4.687    4.687 bpr.py:56(fit)
   500000    3.975    0.000    3.975    0.000 bpr.py:80(_step)
        1    0.000    0.000    0.034    0.034 data.py:18(build_user_item_matrix)
```

500,000 calls to `_step` — one per (positive interaction × epoch). Each
call is a handful of numpy vector ops on tiny 16-dim arrays; the work is
trivial but the Python interpreter overhead per call dominates.

| Bucket                          | Time    | Share |
|---------------------------------|---------|-------|
| `_step` inner loop              | 3.975 s | **85%** |
| `fit` outer loop (resampling, indexing) | 0.674 s | 14%   |
| Data prep (pandas pivot)        | 0.034 s | <1%   |

**Reading:** the `_step` body is too small to vectorize across (it mutates
shared embedding rows in-place), so numpy won't help. The interpreter
itself is the bottleneck. A compiled inner loop is the only intervention
that matters here.

## Where ALS spends its time

```
         401511 function calls in 0.498 seconds

   ncalls  tottime  percall  cumtime  percall function
        1    0.006    0.006    0.498    0.498 als.py:56(fit)
       10    0.304    0.030    0.455    0.046 als.py:75(_solve_side)
    13125    0.077    0.000    0.151    0.000 numpy.linalg.solve
```

`_solve_side` is 91% of ALS fit, but ALS fit is itself 9× *faster* than
BPR. Of `_solve_side`'s time, only one-third is `numpy.linalg.solve` — the
rest is the per-row matrix prep (`cu_minus_1 * other`, the Python loop
itself).

A pre-Phase-1.1 expectation was that ALS would be the obvious Phase 2
target. The numbers say otherwise: ALS is already fast enough that
rewriting it would not be the highest-leverage move.

## What this means for Phase 2

The original plan named ALS as Phase 2's target. The data points elsewhere.

| Candidate           | Speed today | Inner loop characteristics       | Phase 2 priority |
|---------------------|-------------|----------------------------------|------------------|
| **BPR `_step`**     | 4.5 s       | 500k Python calls × tiny vector op | **First**        |
| ALS `_solve_side`   | 0.5 s       | 13k Python calls × 16×16 linalg    | Defer, possibly Phase 3 with sparse work |
| ItemKNN cosine sim  | 0.14 s      | One sklearn call, already C-backed | Skip — no win available |
| Everything else     | < 0.1 s     | Negligible                       | Skip            |

The Phase 2 PR should rewrite `BPR._step` (and the negative-resample loop
that surrounds it) in Rust+PyO3. ALS becomes a Phase 3 candidate *if*
profiling on goodbooks-full — once Phase 3's sparse work makes that
tractable — shows it on the critical path. If it doesn't, we let ALS be.

This is exactly the kind of course-correction Phase 1.1 exists to enable.
