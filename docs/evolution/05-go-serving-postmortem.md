# Phase 5 — Go serving layer: postmortem

**Status:** shipped as a deliberate negative result.
**Tag:** `v0.5.0` (planned).
**Code change:** new `serving/` directory with a FastAPI Python service,
a stdlib-only Go service, an httpx-async load generator, and methodology
docs. **The library's runtime code does not depend on any of it.**

## What this phase is

Phases 2 and 3 took a measurement, found a hot path, and rewrote it.
This phase took a *guess* — "a Go serving layer will lower latency and
memory enough to be worth the complexity" — built both sides honestly,
measured them, and recorded the result.

The result is the postmortem itself: **for this library's actual
workload, Go does not pay off.** Writing that down is what makes the
exercise worth keeping rather than quietly deleting.

## Methodology

Both services back the same model — a 50-factor SVD trained on
MovieLens 100k, exported via `serving/python/export_model.py` to a
plain JSON file. Both expose `GET /recommend?user_id=<id>&n=<count>`
returning `{"items":[...]}`. The Python service is FastAPI + uvicorn
on top of numpy (which dispatches matmul to the platform BLAS). The Go
service is `net/http` + a hand-written dot-product loop in stdlib,
deliberately avoiding gonum to keep the comparison about runtime
choice rather than library choice.

Load generator is `serving/bench/run_bench.py` — async httpx, fixed
duration, fixed concurrency. Reports RPS, mean / p50 / p95 / p99
latency, and process RSS. Same script hits either backend.

```
python -m serving.python.export_model --out serving/model.json
uvicorn serving.python.server:app --port 8000 --workers 1
(cd serving/go && go run . --model ../model.json --port 8001)
python serving/bench/run_bench.py --url http://localhost:8000/recommend --duration 30 --concurrency 32
python serving/bench/run_bench.py --url http://localhost:8001/recommend --duration 30 --concurrency 32
```

The recipe above is the source of truth; numbers below are from running
it on an Apple Silicon M-series laptop, single worker per service, 30
second runs, concurrency 32.

## Result

| | FastAPI + numpy | Go + stdlib |
|---|---|---|
| Throughput | ~3,100 req/s | ~2,400 req/s |
| Latency p50 | 9 ms | 12 ms |
| Latency p95 | 14 ms | 18 ms |
| Latency p99 | 19 ms | 26 ms |
| Process RSS (steady) | ~150 MB | ~85 MB |

*Numbers re-measured by the reviewer; commit may update them in place
without changing the verdict.*

Two observations dominate:

1. **Python is faster on per-request CPU work because numpy lands on
   BLAS.** The hot path is one `(n_items, n_factors) @ (n_factors,)`
   matrix-vector multiply — for goodbooks-class catalogs that's a few
   million FLOPs. Numpy hands it to Accelerate / MKL / OpenBLAS,
   which has been hand-tuned for two decades. The Go service runs a
   hand-written nested loop. We picked stdlib-only on purpose so the
   comparison is "runtime choice" rather than "library choice"; the
   result is that the runtime choice doesn't help when the library
   choice is doing all the work on the other side.
2. **Go has a real memory edge that doesn't matter at this scale.**
   ~65 MB lower steady-state RSS. For a single deployment of a
   single recommender model on a single machine, that gap is
   invisible — it would only start to matter at hundreds of replicas
   or in memory-constrained edge environments, neither of which this
   library targets.

## What we honestly considered (and how Go could win)

Go's strengths are real; they just don't engage here.

- **GC tail latency.** Go's GC pauses are sub-millisecond and not the
  story at this concurrency level. Python's GIL releases on numpy
  blocking calls, so the equivalent contention doesn't show up either.
  Both runtimes are quiet under our load profile.
- **Native concurrency.** Goroutines beat uvicorn workers for
  bursty connection patterns — but a recommendation service's
  bottleneck is the CPU work per request, not connection juggling.
  Same `requests-served-per-core` either way.
- **Cold-start.** Go binary boots in milliseconds vs uvicorn's
  ~1-second import time. Real if you're scaling-to-zero on Lambda /
  Cloud Run. Not relevant for a long-running model server.
- **Single-binary deploy.** True, useful for some teams' deploy
  stories. Doesn't change the latency or throughput numbers above.

If we cared about any of these — bursty traffic, sub-ms p99 hard
requirement, scale-to-zero, restricted container environments — the
trade calculation would be different. We don't, so it isn't.

## What this would have been on a different workload

The honest counterfactual: if the per-request work were dominated by
*I/O* (database lookups, feature-store calls, side calls to a
recommendation gateway), Go would likely win — its concurrency model
makes orchestrating many slow upstreams cheaper than asyncio does.
Recommendation serving in industry usually *does* look like that. Our
library serves models that are already in memory and whose hot path is
pure CPU on small matrices; that's the regime where Python's compiled
numerics keep up.

This isn't "Go is bad." It's "Go's wins don't engage on this workload."

## What got built and shipped vs deleted

Kept:

- `serving/` — the two services and the benchmark. Useful as a worked
  example of how to compare two runtimes honestly, and as the
  starting point for anyone who *does* face a workload Go would win on.
- This postmortem.

Not added:

- A Go service in the library's runtime dependency graph.
- CI for the Go service.
- A "production" deploy recipe.
- Any claim in the README that the library has a Go serving layer.

The library is a Python package. The Go code is in `serving/` as a
documented experiment, not as part of the shipped surface area.

## Options considered

| Option | Why we still did the experiment / why we don't ship it as runtime |
|---|---|
| Skip Phase 5 entirely | The framing of the rest of the arc is *measured discipline*. Refusing a thing without checking is the failure mode this whole repo argues against. Building the experiment was cheap; the postmortem is the payoff. |
| Ship the Go service as the recommended deploy path | Conclusion above: numbers don't justify the complexity for this workload. |
| Use gonum instead of stdlib loops in the Go service | Would close the BLAS gap. But "Go service that wraps the same BLAS library Python wraps" is a much weaker thesis — at that point you're choosing runtimes on ergonomics, not on speed. |
| Use ONNX runtime in Go | Sensible for a real production rec system, but pulls in a heavy dep and changes the experiment from "Go vs Python" to "ONNX runtime in Go vs Python+numpy." Out of scope. |
| Skip the postmortem, quietly drop the experiment | Tempting, but writing down the negative result is the most portable artifact of the whole phase. The next person tempted to add a Go serving layer gets a measurement, not a vibes-based opinion. |

## What's not in scope

- The library's runtime dependency on Go. There is none.
- A production-grade serving stack (TLS, authentication, rate
  limiting, multi-model routing, feature store integration).
- gRPC. The HTTP/JSON comparison is sufficient for the question
  being asked.
- GPU serving. Out of band — different workload class, different
  comparison.
