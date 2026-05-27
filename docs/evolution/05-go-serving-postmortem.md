# Phase 5 — a Go serving layer, and why it beat the Python one

**Status:** shipped as an experiment. The library's runtime does not depend on it.
**Tag:** `v0.5.0` (planned).
**Code:** a new `serving/` directory holding a FastAPI service, a stdlib-only Go
service, a model exporter, and a load generator. Nothing under `serving/` is
imported by `recommender_systems`.

## The bet

Phases 2 and 3 were the same move twice: measure, find the hot path, rewrite it
in something faster. This phase started as a bet that the move had one more place
to go, that putting a trained model behind a small Go HTTP service would serve
recommendations with lower latency and a smaller memory footprint than FastAPI on
numpy.

I expected to lose the latency half of that bet. The per-request work is a single
matrix-vector multiply against the item factors, and numpy hands that to BLAS,
which has had two decades of tuning a hand-written Go loop has no business
beating. The most I expected was a memory win: a static binary holding float32
slices should sit well under a Python process that also carries the interpreter,
FastAPI, and numpy. The plan was to build both, measure, write down that Go bought
some RAM and cost some throughput, and move on.

That is not how it went.

## What actually happened

Both services back the same model, a 50-factor SVD trained on MovieLens 100k,
exported to JSON so the only thing that differs between them is the runtime. Both
serve `GET /recommend?user_id=&n=`. Thirty-second runs, concurrency 32, one worker
per service, on an Apple Silicon laptop (Go 1.26, single uvicorn worker).

| | FastAPI + numpy | Go + stdlib |
|---|---|---|
| Throughput | 585 req/s | 782 req/s |
| Latency p50 | 30 ms | 28 ms |
| Latency p95 | 175 ms | 114 ms |
| Latency p99 | 299 ms | 195 ms |
| Server RSS, steady | 67 MiB | 21 MiB |

Go won all of it: a third more throughput, a third off the p99, a third of the
memory. The one thing I was sure of going in, that BLAS would carry Python on the
CPU work, was simply wrong for this workload.

## Why the BLAS argument didn't hold

The assumption was right about BLAS and wrong about how much of a request BLAS
accounts for. The multiply is 1,682 items by 50 factors, roughly eighty thousand
multiply-adds. On a modern core that takes a few microseconds whether BLAS does it
or a plain `for` loop does, and it is nowhere near the cost of a request.

What a request actually costs is the envelope around the multiply: parsing the
query, the ASGI machinery, allocating the score vector, the top-N selection, JSON
serialization, and Python's per-call overhead layered over all of it. Go does that
same envelope in compiled code with far less allocation. So the benchmark was never
really "BLAS versus a hand loop." It was "the Python request path versus the Go
request path," and at this payload size the request path is the whole game.

To rule out the boring explanation, that Go just used more cores while a
single-process Python sat on one, I pinned Go to one core with `GOMAXPROCS=1` and
ran it again: 687 req/s at 20 MiB, still ahead of Python's 585 at 67. Go is faster
here per core, not only because it has more of them.

## The honest caveat

A single uvicorn worker is the weakest way to run the Python service. The GIL keeps
one process on one core for CPU work, so a lone worker leaves most of the machine
idle, and nobody deploys it that way. The real comparison gives Python
`gunicorn -w <cores>`, and with enough workers its aggregate throughput will pass a
single Go process. It pays for that in memory, though: every worker carries its own
interpreter, its own numpy, and its own copy of the model, so N workers is roughly N
times the 67 MiB. Per-request latency doesn't improve either, since each request
still runs on one worker. Grant Python the deployment it would actually use and it
can win on total throughput, but Go keeps the memory result outright and stays ahead
on latency.

This also only holds while the per-request work stays small. A bigger model, a
larger catalog, or batched scoring would push real FLOPs into the multiply, BLAS
would start to earn its keep, and the picture would drift back toward numpy. The
result is specific to serving a small model with one multiply per request, which
happens to be the shape a lot of recommendation endpoints actually have.

## What this means for the library

Nothing in the package changes. `recommender_systems` is a Python library and stays
one. `pip install` pulls in no Go, CI doesn't build it, and the README claims no
serving runtime. `serving/` stays in the tree as a documented experiment and a
worked example of comparing two runtimes without quietly rigging the result.

What changed is the lesson. Phase 1.1's rule was measure before you cut. This is the
same rule aimed the other way: measure before you dismiss. I was a benchmark away
from shipping a confident writeup about Go not being worth it, and the numbers said
I had the answer backwards. The surprising version is more useful than the tidy one
I planned to write.

## Methodology

Same JSON model for both services, served behind the identical endpoint. The Python
side is FastAPI and uvicorn over numpy; the Go side is `net/http` with a hand-written
dot product in stdlib, no gonum, so the comparison is about the runtime rather than
which BLAS each one wraps. Top-N selection is partial on both sides (`np.argpartition`
in Python, a min-heap in Go) so neither pays for a full sort the other skips. The
load generator (`serving/bench/run_bench.py`) drives async httpx at a fixed duration
and concurrency, reports RPS and client-side p50/p95/p99, and with `--server-pid`
samples the server process RSS via `ps` at the end.

```
python -m serving.python.export_model --out serving/model.json
uvicorn serving.python.server:app --port 8000 &  PY_PID=$!
(cd serving/go && go build -o /tmp/recsvc . && /tmp/recsvc --model ../model.json --port 8001) &  GO_PID=$!
python serving/bench/run_bench.py --url http://localhost:8000/recommend --duration 30 --concurrency 32 --server-pid $PY_PID
python serving/bench/run_bench.py --url http://localhost:8001/recommend --duration 30 --concurrency 32 --server-pid $GO_PID
```

Numbers above are from this recipe on Apple Silicon. They move with hardware; the
gaps are what carry across machines, and they're large enough to survive a fair bit
of variance.

## Options considered

| Option | Verdict |
|---|---|
| Ship the Go service as the library's deploy path | The numbers favor Go for this workload, but the library is a Python package. Carrying a second runtime, its build, and its CI for a serving layer most users won't deploy isn't a trade this project makes. The experiment and the numbers are the deliverable. |
| Use gonum instead of stdlib loops | Would wrap the same BLAS Python wraps, turning the comparison into ergonomics rather than runtime. It also turned out unnecessary: the stdlib loop already won, because the multiply was never the bottleneck. |
| ONNX runtime in Go | Reasonable for a production system, but it pulls in a heavy dependency and changes the question from "Go vs Python" to "ONNX-in-Go vs Python+numpy." Out of scope. |
| Don't write it up | The result is the opposite of what I expected, which makes it the most worth writing down. The next person who assumes BLAS settles this gets a measurement instead of my old guess. |

## Not in scope

- A runtime dependency on Go. There is none.
- A production serving stack: TLS, auth, rate limiting, multi-model routing, a
  feature store.
- gRPC. HTTP/JSON answers the question being asked.
- GPU serving. Different workload class, different comparison.
