"""Structured tracing: per-step spans with trace ids, latency and token/cost estimates.

Traces are persisted to SQLite (see storage.py), emitted as JSON log lines and, when
``OTEL_EXPORTER_OTLP_ENDPOINT`` is set, pushed to an OpenTelemetry collector over OTLP/HTTP JSON.
Stored questions are always PII-redacted.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx

from .resilience import CircuitBreaker

log = logging.getLogger("rag_guardian.trace")

SPAN_NAMES = ("guardrail_in", "retrieve", "rerank", "generate", "guardrail_out")
_MAX_FIELD_CHARS = 12_000


def _clip(value: Any) -> Any:
    try:
        raw = json.dumps(value, default=str)
    except (TypeError, ValueError):
        return {"_unserialisable": str(value)[:500]}
    if len(raw) <= _MAX_FIELD_CHARS:
        return json.loads(raw)
    return {"_truncated": True, "preview": raw[:2000]}


@dataclass
class Span:
    name: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    start_unix: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    status: str = "ok"  # ok | blocked | error | degraded
    input: Any = None
    output: Any = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "span_id": self.span_id, "start_unix": round(self.start_unix, 6),
            "duration_ms": round(self.duration_ms, 3), "status": self.status,
            "input": _clip(self.input), "output": _clip(self.output), "attrs": _clip(self.attrs),
        }


class Trace:
    def __init__(self, user_id: str, roles: str, trace_id: str | None = None):
        self.trace_id = trace_id or uuid.uuid4().hex
        self.user_id = user_id
        self.roles = roles
        self.created_at = datetime.now(UTC).isoformat(timespec="milliseconds")
        self.started = time.perf_counter()
        self.spans: list[Span] = []
        self.tokens_in = 0
        self.tokens_out = 0
        self.cost_usd = 0.0
        self.question = ""
        self.status = "pending"
        self.reason = ""
        self.confidence = 0.0
        self.degraded: list[str] = []
        self.final_answer: str | None = None
        self.citations: list[dict[str, Any]] = []
        self.latency_ms = 0.0

    @contextmanager
    def span(self, name: str) -> Iterator[Span]:
        sp = Span(name)
        t0 = time.perf_counter()
        try:
            yield sp
        except Exception as exc:
            sp.status = "error"
            sp.attrs["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            sp.duration_ms = (time.perf_counter() - t0) * 1000
            self.spans.append(sp)
            log.info(json.dumps({"event": "span", "trace_id": self.trace_id, "span": sp.name, "status": sp.status,
                                 "duration_ms": round(sp.duration_ms, 3)}))

    def finish(self) -> None:
        self.latency_ms = (time.perf_counter() - self.started) * 1000
        log.info(json.dumps({"event": "trace", "trace_id": self.trace_id, "status": self.status, "reason": self.reason,
                             "latency_ms": round(self.latency_ms, 3), "tokens_in": self.tokens_in,
                             "tokens_out": self.tokens_out, "cost_usd": round(self.cost_usd, 6)}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id, "created_at": self.created_at, "user_id": self.user_id, "roles": self.roles,
            "question": self.question, "status": self.status, "reason": self.reason,
            "latency_ms": round(self.latency_ms, 3), "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
            "cost_usd": round(self.cost_usd, 6), "confidence": round(self.confidence, 3), "degraded": self.degraded,
            "final_answer": self.final_answer, "citations": self.citations, "spans": [s.to_dict() for s in self.spans],
        }


def configure_json_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.handlers[:] = [handler]
    log.setLevel(level)
    log.propagate = False


def _attr(key: str, value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        v: dict[str, Any] = {"boolValue": value}
    elif isinstance(value, int):
        v = {"intValue": str(value)}
    elif isinstance(value, float):
        v = {"doubleValue": value}
    else:
        v = {"stringValue": str(value)}
    return {"key": key, "value": v}


def to_otlp(trace: dict[str, Any], service_name: str = "policy-rag-guardian") -> dict[str, Any]:
    """Convert a stored trace to an OTLP/HTTP JSON ``ExportTraceServiceRequest`` body."""
    start = min((s["start_unix"] for s in trace["spans"]), default=time.time())
    root_id = uuid.uuid4().hex[:16]
    ns = lambda t: str(int(t * 1e9))  # noqa: E731
    spans = [{
        "traceId": trace["trace_id"], "spanId": root_id, "name": "rag.query", "kind": 2,
        "startTimeUnixNano": ns(start), "endTimeUnixNano": ns(start + trace["latency_ms"] / 1000),
        "attributes": [_attr("rag.status", trace["status"]), _attr("rag.reason", trace["reason"]),
                       _attr("rag.confidence", trace["confidence"]), _attr("rag.tokens_in", trace["tokens_in"]),
                       _attr("rag.tokens_out", trace["tokens_out"]), _attr("rag.cost_usd", trace["cost_usd"])],
        "status": {"code": 1},
    }]
    for s in trace["spans"]:
        spans.append({
            "traceId": trace["trace_id"], "spanId": s["span_id"], "parentSpanId": root_id, "name": s["name"], "kind": 1,
            "startTimeUnixNano": ns(s["start_unix"]), "endTimeUnixNano": ns(s["start_unix"] + s["duration_ms"] / 1000),
            "attributes": [_attr("rag.span.status", s["status"])] + [_attr(f"rag.{k}", v) for k, v in s["attrs"].items()
                                                                        if isinstance(v, str | int | float | bool)],
            "status": {"code": 2 if s["status"] == "error" else 1},
        })
    return {"resourceSpans": [{
        "resource": {"attributes": [_attr("service.name", service_name)]},
        "scopeSpans": [{"scope": {"name": "rag_guardian"}, "spans": spans}],
    }]}


class OtlpExporter:
    """Best-effort background exporter. Never raises into the request path; has its own breaker."""

    def __init__(self, endpoint: str, client: httpx.Client | None = None, synchronous: bool = False):
        self.endpoint = endpoint.rstrip("/") + "/v1/traces"
        self._client = client or httpx.Client(timeout=3.0)
        self._pool = None if synchronous else ThreadPoolExecutor(max_workers=1, thread_name_prefix="otlp")
        self.breaker = CircuitBreaker("otlp", failure_threshold=3, recovery_timeout=30)
        self.sent = 0
        self.failed = 0

    def export(self, trace: dict[str, Any]) -> None:
        if self._pool:
            self._pool.submit(self._send, trace)
        else:
            self._send(trace)

    def _send(self, trace: dict[str, Any]) -> None:
        if not self.breaker.allow():
            return
        try:
            self._client.post(self.endpoint, json=to_otlp(trace)).raise_for_status()
            self.breaker.record_success()
            self.sent += 1
        except Exception:  # noqa: BLE001
            self.breaker.record_failure()
            self.failed += 1
