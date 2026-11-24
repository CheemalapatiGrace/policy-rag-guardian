import asyncio
import time

import pytest

from rag_guardian.resilience import (
    CallTimeoutError,
    CircuitBreaker,
    CircuitOpenError,
    Resilient,
    aretry,
    backoff_delay,
    call_with_timeout,
    retry,
)


def test_backoff_is_exponential_capped_and_jittered():
    assert [backoff_delay(a, 0.1, 1.0, rng=lambda: 1.0) for a in range(5)] == [0.1, 0.2, 0.4, 0.8, 1.0]
    assert backoff_delay(3, 0.1, 10.0, rng=lambda: 0.0) == 0.0  # full jitter can be zero
    assert 0 <= backoff_delay(2, 0.1, 10.0) <= 0.4


def test_retry_succeeds_after_transient_failures_with_backoff_sleeps():
    calls, sleeps = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("boom")
        return "ok"

    assert retry(flaky, attempts=4, base_delay=0.1, sleep=sleeps.append, rng=lambda: 1.0) == "ok"
    assert len(calls) == 3 and sleeps == [0.1, 0.2]


def test_retry_raises_last_error_when_exhausted():
    with pytest.raises(ValueError, match="nope"):
        retry(lambda: (_ for _ in ()).throw(ValueError("nope")), attempts=2, sleep=lambda s: None)


def test_retry_does_not_retry_open_circuit():
    n = []

    def f():
        n.append(1)
        raise CircuitOpenError("open")

    with pytest.raises(CircuitOpenError):
        retry(f, attempts=5, sleep=lambda s: None)
    assert len(n) == 1


def test_async_retry():
    state = {"n": 0}

    async def f():
        state["n"] += 1
        if state["n"] < 2:
            raise OSError("x")
        return state["n"]

    async def nosleep(_):
        return None

    assert asyncio.run(aretry(f, attempts=3, sleep=nosleep)) == 2


def test_circuit_breaker_opens_half_opens_and_closes():
    t = [0.0]
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=10, clock=lambda: t[0])
    assert cb.state == "closed" and cb.allow()
    cb.record_failure()
    cb.record_failure()
    assert cb.state == "open" and not cb.allow()
    t[0] = 11
    assert cb.state == "half_open" and cb.allow() and not cb.allow()  # single probe only
    cb.record_success()
    assert cb.state == "closed" and cb.allow()


def test_failed_probe_reopens_breaker():
    t = [0.0]
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=5, clock=lambda: t[0])
    cb.record_failure()
    t[0] = 6
    assert cb.allow()
    cb.record_failure()
    assert cb.state == "open"


def test_resilient_trips_breaker_and_fails_fast():
    r = Resilient("dep", attempts=2, sleep=lambda s: None, timeout=1, breaker=CircuitBreaker(failure_threshold=2))

    def bad():
        raise ConnectionError("down")

    for _ in range(2):
        with pytest.raises(ConnectionError):
            r.call(bad)
    with pytest.raises(CircuitOpenError):
        r.call(bad)
    assert r.stats()["state"] == "open" and r.retries == 2


def test_timeout_abandons_slow_calls():
    with pytest.raises(CallTimeoutError):
        call_with_timeout(time.sleep, 0.05, 0.5)
    assert call_with_timeout(lambda: 7, 1.0) == 7
