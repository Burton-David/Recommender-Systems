# Phase 2 — Rust+PyO3 kernel for BPR

**Status:** shipped.
**Tag:** `v0.2.0` (planned).
**Code change:** new `crates/recsys-kernels/` crate, build switched from
Hatchling to maturin, `BPR.fit` calls the kernel with a pure-Python fallback.

## Context

The [Phase 1.1 baseline](01-1-bench-baseline.md) measured BPR fit at 4.5 s on
MovieLens 100k — an order of magnitude slower than anything else in the
library. `cProfile` traced 85% of that to a single function: `BPR._step`,
called 500,000 times per epoch budget. The work inside each call is small —
three vector subtractions, a dot product, a sigmoid, three in-place updates on
16-dimensional rows — but the Python interpreter overhead per call dominates.

Numpy can't help here. The body is too small for SIMD to matter and the
updates mutate shared embedding rows, so there's nothing to vectorize across.
The interpreter itself is the bottleneck.

## Decision

Rewrite the BPR training loop — `_step` plus the negative-resample logic
around it — in Rust, exposed to Python via PyO3. Everything else stays in
Python: data loading, matrix construction, factor initialization, the public
`fit` signature, the `recommend` path.

The build moves from Hatchling to maturin so the same `pip install -e .`
flow compiles the extension and drops a `.so` into `recommender_systems/`.
Wheels published to PyPI (a future phase) will be pre-built per platform via
`cibuildwheel`; end users never see Rust.

## Options considered

| Option | Rejected because |
|---|---|
| Vectorize `_step` across positives in numpy | The update mutates shared rows; you'd lose the SGD semantics and converge to a different objective. |
| Cython | Same end result as Rust for our workload but the ecosystem is shrinking — polars, pydantic-core, tokenizers, ruff all moved off it. New libraries reaching for compiled kernels in 2026 reach for PyO3. |
| `numba` JIT | Lighter ramp than Rust but adds an LLVM dependency and a one-time JIT cost on first call. Numba's `@jit` on the existing `_step` would help; pure-Rust would help more and avoids the LLVM dep. |
| C extension via cffi/cython-free | Faster to write but unsafe and brittle. A segfault in a C extension takes down the interpreter. Rust's safety guarantees survive ten years of contributors. |
| Pre-sample negatives in numpy, do gradient updates only in Rust | Considered for bit-equivalence with the Python loop. Rejected because the resample-while-collision pattern is hot itself; pushing it to Rust along with `_step` is cleaner. We trade bit-equivalence for a single self-contained Rust function. |
| Keep the Python loop, accept BPR is slow | Tempting, but the Phase 1.1 measurements identify BPR as the one place where the library noticeably stalls. Closing the gap is what Phase 2 exists to do. |

## What's in the kernel

`crates/recsys-kernels/src/lib.rs` exposes one function:

```rust
fn bpr_train(
    user_factors, item_factors,   // mutated in place
    positives,                    // (n_pos, 2) int64
    observed_flat,                // row-major (n_users * n_items,) bool
    n_items, epochs, learning_rate, reg, seed,
) -> PyResult<()>
```

Internals:

- Fisher–Yates shuffle of the positive index list once per epoch.
- For each shuffled positive: uniform-random negative sample with a resample
  loop until `observed[u, j]` is false.
- The gradient update is the same sigmoid-margin form `BPR._step` has
  always implemented, with the same regularization term.
- RNG is PCG64 (`rand_pcg`) seeded by the `random_state` Python passes in.

Performance numbers (release build on Apple-Silicon):

| | Before (Python) | After (Rust kernel) | Speedup |
|---|---|---|---|
| BPR fit, ML-100k, 5 epochs | 4,503 ms | ~88 ms | ~51× |

Hardware-dependent; the speedup ratio is what carries. The committed
`tests/test_fit_speed.py::test_bpr_fit` benchmark now exercises the kernel
path, so the suite records a fresh number every time it runs. Full table
in `benchmarks/profile.md`.

## What's not in scope

- **ALS.** Phase 1.1 showed ALS is already 9× faster than BPR. Its
  `_solve_side` is 91% of its time, but its time is small. Phase 3
  ("sparse-aware everywhere") will revisit ALS if profiling on
  goodbooks-full puts it back on the critical path.
- **`recommend()` hot path.** Top-N selection over the dense score vector is
  fast enough at MovieLens scale; the win there only matters once Phase 3
  unlocks larger catalogs.
- **Bit-equivalence with the Python loop.** Two independent reasons the
  float arrays diverge:
  *(a)* the paths consume RNG bytes in different orders, so the permutations
  and negative samples won't match; and
  *(b)* the kernel snapshots the user row before any update so the positive
  and negative item updates use the *pre-update* user vector (textbook
  simultaneous SGD), whereas `BPR._step` mutates `self._user_factors[u]`
  in place and the subsequent item updates inherit the *post-update* user
  via numpy view aliasing. The kernel's formulation is the standard one;
  the Python aliasing is a quirk of how the in-place updates were written.
  Regression tests assert *recommendation-quality* equivalence —
  precision@10 and recall@10 must clear conservative floors on the
  canonical split — instead of bit-equivalence.
- **Hogwild-style parallel SGD.** Plausible follow-up if a single-machine
  training bottleneck appears. Skipped now because (a) it adds
  non-determinism, and (b) the sequential Rust loop already closes the
  Phase 1.1 gap.

## Falling back to Python

`BPR.fit` tries `from recommender_systems import _kernels`; on `ImportError`
it runs the original pure-Python loop. This keeps the library usable in
environments where the compiled extension didn't ship — sdist installs
without a Rust toolchain, exotic platforms not covered by
`cibuildwheel`, etc.

CI builds the extension on every leg, so the kernel path is always exercised.
The fallback path is tested by `tests/test_bpr_fallback.py`, which suppresses
the `_kernels` import via `sys.modules` before importing `BPR` and runs the
same training contract end-to-end — same CI run, no extra job.
