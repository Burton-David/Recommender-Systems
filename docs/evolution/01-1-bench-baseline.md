# Phase 1.1 — measure before cutting

**Status:** shipped.
**Tag:** `v0.1.1` (planned).
**Code change:** none in `src/`. New `tests/benchmarks/` suite and
`benchmarks/profile.md`.

## Context

The library has reached the "useful but unoptimized" state. Several
optimization directions are obvious in principle — Rust kernels, sparse
matrices, GPU. Picking which one to do *first* requires evidence, and the
repo has none. There are committed benchmark *result tables* in
`benchmarks/results.md`, but those measure recommendation quality, not
where wall-clock time goes during training.

## Decision

Ship a benchmark + profile baseline as its own phase, before any rewriting.
Three deliverables:

1. A `pytest-benchmark` suite under `tests/benchmarks/` covering fit time
   for every algorithm in the library on MovieLens 100k.
2. A `benchmarks/profile.md` document that records, for the slow ones:
   a `cProfile` breakdown showing which function consumes the time.
3. A `not benchmark` filter in `pyproject.toml` so the timing suite is
   opt-in (it's wall-clock-sensitive; CI on shared runners would be noisy).

## Options considered

| Option | Rejected because |
|---|---|
| Skip Phase 1.1, start Rust rewrite of ALS | Without numbers, "ALS is the hot path" is an opinion. The whole rewrite plan rests on knowing which inner loop actually dominates. |
| Use `timeit` ad-hoc, don't commit a suite | Reproducibility matters. A reviewer should be able to run the suite and see whether the claims hold. Ad-hoc timings rot. |
| Run benchmarks in CI | Wall-clock numbers on GitHub Actions are too noisy to gate on. The committed baseline is the source of truth; CI verifies correctness, not speed. |
| Use `asv` (airspeed velocity) instead of `pytest-benchmark` | `asv` is the right tool when you want a time-series of regression detection across commits. `pytest-benchmark` is right when you want one snapshot per phase. We have phases, not a continuous regression budget. |
| Profile in production with `py-spy` flamegraphs | Worth doing later; cProfile's function-level view is enough to answer "which function is the hot path." |

## The finding that mattered

The original plan named ALS's `_solve_side` as Phase 2's Rust target.
The measurements say otherwise:

- **BPR fit: 4,503 ms.** `_step` is 85% of the time. 500,000 Python
  function calls on tiny vector ops; the interpreter is the bottleneck.
- **ALS fit: 473 ms.** `_solve_side` is 91% of *that* but ALS as a whole
  is 9× *faster* than BPR.

Phase 2 will rewrite BPR, not ALS. ALS becomes a Phase 3 candidate if
sparse work on goodbooks-full puts it on the critical path.

This course-correction is the *reason* Phase 1.1 exists. If the data had
confirmed the original guess we'd have proceeded; instead it pointed at
a better target.

## What's not in scope

- No optimization of any algorithm — that's Phase 2 onward.
- No flamegraphs. `cProfile` answers the question being asked
  (which function is the hot path?). Flamegraphs are richer but slower
  to read and to commit; defer until we need them.
- No `goodbooks-10k` benchmarks. The dataset requires subsampling for the
  dense algorithms, which would bias the comparison. Phase 3 lands sparse
  variants; *then* the full corpus becomes the benchmark frame.

## Results

See `benchmarks/profile.md` for the table and profile output.
