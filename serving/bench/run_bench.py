"""Apples-to-apples load generator for the Python and Go recommendation services.

Runs concurrent GET /recommend requests for a fixed duration, then reports
RPS plus a latency histogram. The same script hits either backend — the only
thing that differs between runs is the ``--url`` argument.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import resource
import time
from collections.abc import Iterable

import httpx


def percentiles(samples: list[float], qs: Iterable[float]) -> dict[float, float]:
    samples_sorted = sorted(samples)
    out: dict[float, float] = {}
    for q in qs:
        idx = min(len(samples_sorted) - 1, max(0, int(q * len(samples_sorted))))
        out[q] = samples_sorted[idx]
    return out


async def worker(
    client: httpx.AsyncClient,
    url: str,
    user_ids: list[int],
    deadline: float,
    latencies: list[float],
    errors: list[int],
) -> None:
    i = 0
    while time.monotonic() < deadline:
        uid = user_ids[i % len(user_ids)]
        i += 1
        t0 = time.monotonic()
        try:
            r = await client.get(url, params={"user_id": uid, "n": 10})
            if r.status_code != 200:
                errors.append(r.status_code)
                continue
        except httpx.HTTPError:
            errors.append(0)
            continue
        latencies.append((time.monotonic() - t0) * 1000.0)


async def main_async(args: argparse.Namespace) -> int:
    user_ids = list(range(1, 944))  # MovieLens 100k user ids run 1..943
    deadline = time.monotonic() + args.duration
    latencies: list[float] = []
    errors: list[int] = []
    timeout = httpx.Timeout(args.timeout)
    limits = httpx.Limits(max_connections=args.concurrency * 2)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        await asyncio.gather(
            *(
                worker(client, args.url, user_ids, deadline, latencies, errors)
                for _ in range(args.concurrency)
            )
        )

    total = len(latencies) + len(errors)
    if total == 0:
        print("no requests completed")
        return 1
    rps = total / args.duration
    pcts = percentiles(latencies, [0.50, 0.95, 0.99])
    mean = sum(latencies) / len(latencies) if latencies else float("nan")
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss_kb / 1024 if os.uname().sysname == "Linux" else rss_kb / (1024 * 1024)

    print(f"url:           {args.url}")
    print(f"duration:      {args.duration}s   concurrency: {args.concurrency}")
    print(f"requests:      {total} ({rps:.1f} rps)")
    print(f"errors:        {len(errors)}")
    print(
        f"latency ms:    mean {mean:.2f}  p50 {pcts[0.50]:.2f}  "
        f"p95 {pcts[0.95]:.2f}  p99 {pcts[0.99]:.2f}"
    )
    print(f"client rss:    {rss_mb:.1f} MiB")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="run_bench")
    parser.add_argument("--url", required=True)
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=2.0)
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
