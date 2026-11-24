import json

import httpx
import pytest

from rag_guardian.evals import __main__ as cli
from rag_guardian.evals.probes import run_probes
from rag_guardian.evals.runner import (
    DEFAULT_THRESHOLDS,
    compute_metrics,
    evaluate_thresholds,
    load_cases,
    run_cases,
)
from rag_guardian.tracing import OtlpExporter, Trace, to_otlp


def test_golden_dataset_is_large_diverse_and_wellformed(settings):
    cases = load_cases(settings.golden_path)
    cats = {c["category"] for c in cases}
    assert len(cases) >= 30 and {"factual", "injection", "acl_bypass", "out_of_scope", "unanswerable", "pii"} <= cats
    assert len({c["id"] for c in cases}) == len(cases)
    assert all(c["expect"] in {"answer", "refuse", "block", "review"} for c in cases)
    assert sum(c["category"] == "injection" for c in cases) >= 8


def test_eval_suite_meets_ci_gates(pipeline, settings):
    results = run_cases(pipeline, load_cases(settings.golden_path))
    metrics = compute_metrics(pipeline, results)
    gates = evaluate_thresholds(metrics)
    assert [g["metric"] for g in gates if not g["passed"]] == []
    assert metrics["acl_leaks"] == 0 and metrics["pii_leaks"] == 0 and metrics["injection_block_rate"] == 1.0


def test_guard_probes_catch_every_hostile_output():
    r = run_probes()
    assert r["failures"] == [] and r["total"] >= 15


def test_threshold_logic_directions():
    m = {k: (0.5 if op == ">=" else 0) for k, (op, _) in DEFAULT_THRESHOLDS.items()}
    failed = {g["metric"] for g in evaluate_thresholds(m) if not g["passed"]}
    assert "recall_at_k" in failed and "acl_leaks" not in failed
    assert all(g["passed"] for g in evaluate_thresholds(m, {k: 0.1 for k, (op, _) in DEFAULT_THRESHOLDS.items() if op == ">="}))


def test_eval_cli_exit_codes_and_json(capsys):
    assert cli.main(["--no-regression"]) == 0
    assert "RESULT: PASS" in capsys.readouterr().out
    assert cli.main(["--no-regression", "--threshold", "recall_at_k=1.01", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is False and not next(g for g in payload["gates"] if g["metric"] == "recall_at_k")["passed"]
    assert cli.main(["--threshold", "bogus=1"]) == 2


def test_eval_gate_catches_a_broken_guardrail(monkeypatch, capsys):
    """Sanity: if the injection guard stops working, the CI gate must go red."""
    from rag_guardian import guardrails

    monkeypatch.setattr(guardrails.InputGuard, "check", lambda self, q: guardrails.InputVerdict(True, None, q))
    assert cli.main(["--no-regression"]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_trace_spans_capture_errors_and_timing():
    t = Trace("u", "employee")
    with pytest.raises(ValueError):
        with t.span("retrieve"):
            raise ValueError("bad")
    assert t.spans[0].status == "error" and "ValueError" in t.spans[0].attrs["error"] and t.spans[0].duration_ms >= 0


def test_oversized_span_payloads_are_truncated():
    t = Trace("u", "employee")
    with t.span("generate") as sp:
        sp.output = {"text": "x" * 50_000}
    assert t.to_dict()["spans"][0]["output"]["_truncated"] is True


def test_otlp_payload_shape_and_export(pipeline):
    from conftest import EMPLOYEE

    r = pipeline.answer("How long are security logs retained?", EMPLOYEE)
    trace = pipeline.db.get_trace(r.trace_id)
    body = to_otlp(trace)
    spans = body["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert len(spans) == 1 + len(trace["spans"]) and all(s["traceId"] == trace["trace_id"] for s in spans)
    assert spans[0]["name"] == "rag.query" and {s["parentSpanId"] for s in spans[1:]} == {spans[0]["spanId"]}

    seen = []
    exp = OtlpExporter("http://collector:4318", httpx.Client(transport=httpx.MockTransport(lambda req: seen.append(req) or httpx.Response(200))), synchronous=True)
    exp.export(trace)
    assert seen[0].url.path == "/v1/traces" and exp.sent == 1


def test_otlp_exporter_never_raises_and_trips_its_own_breaker():
    def boom(req):
        raise httpx.ConnectError("collector down")

    exp = OtlpExporter("http://x", httpx.Client(transport=httpx.MockTransport(boom)), synchronous=True)
    for _ in range(5):
        exp.export({"trace_id": "a" * 32, "spans": [], "status": "ok", "reason": "", "confidence": 0, "tokens_in": 0,
                    "tokens_out": 0, "cost_usd": 0, "latency_ms": 1})
    assert exp.failed == 3 and exp.breaker.state == "open"  # stopped hammering after 3 failures
