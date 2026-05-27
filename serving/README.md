# Serving layer experiment

This directory holds two recommendation-serving implementations — one in Go,
one in Python (FastAPI + numpy) — set up so the same model can be served by
either and benchmarked apples-to-apples. The point of the exercise is the
[Phase 5 postmortem](../docs/evolution/05-go-serving-postmortem.md):
*does a Go serving layer actually pay off for this library's workload?*

The answer is: not really. The postmortem walks through why.

## What's here

```
serving/
├── python/         FastAPI service that loads a saved SVD model and serves /recommend
├── go/             Go HTTP service over the same model file, same endpoint
├── bench/          httpx-async load generator that hits either service
└── README.md       this file
```

The model format is a plain JSON blob (`model.json`) with three arrays:

```json
{
  "user_ids":      [1, 2, 3, ...],
  "item_ids":      ["a", "b", "c", ...],
  "user_factors":  [[...], [...], ...],
  "item_factors":  [[...], [...], ...]
}
```

JSON is deliberately the lowest-common-denominator format so the two
implementations can be compared without arguing about serialization
overhead — both pay the same parse cost at startup.

## Exporting a model

From the project root:

```bash
python -m serving.python.export_model --algo svd --out serving/model.json
```

That trains `recommender_systems.svd.SVD(n_factors=50)` on MovieLens 100k
and writes the factors out as JSON.

## Running each service

### Python (FastAPI)

```bash
pip install -e ".[neural]" fastapi 'uvicorn[standard]' httpx
uvicorn serving.python.server:app --host 0.0.0.0 --port 8000 --workers 1
```

### Go

```bash
cd serving/go
go run . --model ../model.json --port 8001
```

Both expose `GET /recommend?user_id=<id>&n=<count>` and return `{"items":[...]}`.

## Running the benchmark

```bash
python serving/bench/run_bench.py --url http://localhost:8000/recommend --duration 30 --concurrency 32
python serving/bench/run_bench.py --url http://localhost:8001/recommend --duration 30 --concurrency 32
```

Reports requests-per-second, mean / p50 / p95 / p99 latency, and process
RSS at the end.

## Results

See [`docs/evolution/05-go-serving-postmortem.md`](../docs/evolution/05-go-serving-postmortem.md)
— it's the deliverable, not an afterthought. Short version: I expected
numpy's BLAS to carry Python and was wrong. The per-request multiply is
tiny, so the request path decides it, not the matrix math, and Go's
request path is leaner. The Go service beat FastAPI + numpy on
throughput, latency, and memory, and still won pinned to a single core.
Multi-worker Python can take back aggregate throughput, at several times
the memory.
