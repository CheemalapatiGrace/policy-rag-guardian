#!/usr/bin/env python
"""Load / latency benchmark: p50/p95/p99 latency and throughput.

In-process (default):   python scripts/bench.py --requests 500 --concurrency 8
Against a live server:  RAG_RATE_LIMIT_PER_MIN=100000 RAG_RATE_LIMIT_BURST=1000 make run   (other shell)
                        python scripts/bench.py --url http://127.0.0.1:8000 --api-key key-employee
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from rag_guardian.auth import Principal
from rag_guardian.config import Settings
from rag_guardian.pipeline import build_pipeline
from rag_guardian.storage import Database


def questions() -> list[str]:
    cases = [json.loads(line) for line in Settings().golden_path.read_text().splitlines() if line.strip()]
    return [c["question"] for c in cases if c["role"] in ("employee", "guest")]


def pct(sorted_ms: list[float], q: float) -> float:
    return sorted_ms[min(len(sorted_ms) - 1, int(q * len(sorted_ms)))]


def summarize(label: str, latencies_ms: list[float], wall_s: float, errors: int, extra: dict | None = None) -> dict:
    lat = sorted(latencies_ms)
    out = {
        "mode": label, "requests": len(lat) + errors, "errors": errors, "wall_seconds": round(wall_s, 3),
        "throughput_rps": round(len(lat) / wall_s, 1) if wall_s else 0.0,
        "latency_ms": {"mean": round(statistics.fmean(lat), 2), "p50": round(pct(lat, 0.50), 2),
                       "p95": round(pct(lat, 0.95), 2), "p99": round(pct(lat, 0.99), 2), "max": round(lat[-1], 2)},
    }
    out.update(extra or {})
    return out


def bench_inprocess(n: int, concurrency: int) -> dict:
    qs = questions()
    with tempfile.TemporaryDirectory() as tmp:
        s = replace(Settings.from_env(), db_path=Path(tmp) / "bench.db", regression_path=Path(tmp) / "reg.jsonl")
        pipe = build_pipeline(s, db=Database(s.db_path))
        who = Principal("bench", frozenset({"employee"}))
        for q in qs[:5]:  # warm-up
            pipe.answer(q, who)

        def one(i: int) -> tuple[float, str]:
            t0 = time.perf_counter()
            r = pipe.answer(qs[i % len(qs)], who)
            return (time.perf_counter() - t0) * 1000, r.status

        t0 = time.perf_counter()
        with ThreadPoolExecutor(concurrency) as ex:
            res = list(ex.map(one, range(n)))
        wall = time.perf_counter() - t0
        statuses: dict[str, int] = {}
        for _, st in res:
            statuses[st] = statuses.get(st, 0) + 1
        return summarize(f"in-process x{concurrency}", [r[0] for r in res], wall, 0,
                         {"statuses": statuses, "embedding_cache": pipe.embedder.stats(), "index": pipe.index.stats()})


async def bench_http(url: str, key: str, n: int, concurrency: int) -> dict:
    import httpx

    qs = questions()
    sem = asyncio.Semaphore(concurrency)
    lat: list[float] = []
    codes: dict[int, int] = {}
    async with httpx.AsyncClient(base_url=url, timeout=30.0, headers={"X-API-Key": key}) as client:
        async def one(i: int) -> None:
            async with sem:
                t0 = time.perf_counter()
                r = await client.post("/v1/query", json={"question": qs[i % len(qs)]})
                ms = (time.perf_counter() - t0) * 1000
                codes[r.status_code] = codes.get(r.status_code, 0) + 1
                if r.status_code == 200:
                    lat.append(ms)

        t0 = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(n)))
        wall = time.perf_counter() - t0
    if not lat:
        raise SystemExit(f"no successful responses: {codes}")
    return summarize(f"http x{concurrency}", lat, wall, n - len(lat), {"http_status_counts": codes})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--requests", type=int, default=500)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--url", help="benchmark a running server instead of in-process")
    ap.add_argument("--api-key", default="key-employee")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    out = asyncio.run(bench_http(a.url, a.api_key, a.requests, a.concurrency)) if a.url else bench_inprocess(a.requests, a.concurrency)
    if a.json:
        print(json.dumps(out, indent=2))
    else:
        lt = out["latency_ms"]
        print(f"\nmode        : {out['mode']}\nrequests    : {out['requests']} ({out['errors']} errors)\n"
              f"wall time   : {out['wall_seconds']} s\nthroughput  : {out['throughput_rps']} req/s\n"
              f"latency ms  : mean {lt['mean']}  p50 {lt['p50']}  p95 {lt['p95']}  p99 {lt['p99']}  max {lt['max']}")
        for k in ("statuses", "http_status_counts", "embedding_cache"):
            if k in out:
                print(f"{k:<12}: {out[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
