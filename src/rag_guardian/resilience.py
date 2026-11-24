"""Reliability primitives: retries with exponential backoff + full jitter, timeouts, circuit breaker.

All time sources (clock, sleep, rng) are injectable so behaviour is unit-testable without waiting.
"""

from __future__ import annotations

import asyncio
import random
import threading
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any, TypeVar

T = TypeVar("T")


class CircuitOpenError(RuntimeError):
    """Raised instead of calling a dependency whose breaker is open."""


class CallTimeoutError(TimeoutError):
    pass


class CircuitBreaker:
    """closed -> (N consecutive failures) -> open -> (recovery_timeout) -> half-open -> closed/open."""

    def __init__(self, name: str = "dep", failure_threshold: int = 5, recovery_timeout: float = 30.0,
                 clock: Callable[[], float] = time.monotonic):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._clock = clock
        self._lock = threading.Lock()
        self._failures = 0
        self._opened_at: float | None = None
        self._probe_in_flight = False

    @property
    def state(self) -> str:
        with self._lock:
            return self._state_locked()

    def _state_locked(self) -> str:
        if self._opened_at is None:
            return "closed"
        if self._clock() - self._opened_at >= self.recovery_timeout:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        with self._lock:
            st = self._state_locked()
            if st == "closed":
                return True
            if st == "half_open" and not self._probe_in_flight:
                self._probe_in_flight = True  # exactly one probe request
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None
            self._probe_in_flight = False

    def record_failure(self) -> None:
        with self._lock:
            self._probe_in_flight = False
            self._failures += 1
            if self._failures >= self.failure_threshold or self._opened_at is not None:
                self._opened_at = self._clock()


_POOL = ThreadPoolExecutor(max_workers=32, thread_name_prefix="rag-timeout")


def call_with_timeout(fn: Callable[..., T], timeout: float | None, *args: Any, **kwargs: Any) -> T:
    """Run ``fn`` in a worker thread and give up waiting after ``timeout`` seconds.

    Python cannot kill a thread, so the abandoned call may finish in the background; callers should
    pair this with idempotent operations (embedding / generation are).
    """
    if timeout is None:
        return fn(*args, **kwargs)
    fut = _POOL.submit(fn, *args, **kwargs)
    try:
        return fut.result(timeout=timeout)
    except FutureTimeout as exc:
        fut.cancel()
        raise CallTimeoutError(f"call exceeded {timeout}s") from exc


def backoff_delay(attempt: int, base: float, cap: float, rng: Callable[[], float] = random.random) -> float:
    """Full-jitter exponential backoff: U(0, min(cap, base * 2**attempt))."""
    return rng() * min(cap, base * (2**attempt))


def retry(fn: Callable[[], T], *, attempts: int = 3, base_delay: float = 0.1, max_delay: float = 2.0,
          retry_on: tuple[type[BaseException], ...] = (Exception,),
          no_retry: tuple[type[BaseException], ...] = (CircuitOpenError,),
          sleep: Callable[[float], None] = time.sleep, rng: Callable[[], float] = random.random,
          on_retry: Callable[[int, BaseException, float], None] | None = None) -> T:
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except no_retry:
            raise
        except retry_on as exc:
            last = exc
            if attempt == attempts - 1:
                break
            delay = backoff_delay(attempt, base_delay, max_delay, rng)
            if on_retry:
                on_retry(attempt + 1, exc, delay)
            sleep(delay)
    assert last is not None
    raise last


async def aretry(fn: Callable[[], Awaitable[T]], *, attempts: int = 3, base_delay: float = 0.1,
                 max_delay: float = 2.0, retry_on: tuple[type[BaseException], ...] = (Exception,),
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 rng: Callable[[], float] = random.random) -> T:
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return await fn()
        except CircuitOpenError:
            raise
        except retry_on as exc:
            last = exc
            if attempt == attempts - 1:
                break
            await sleep(backoff_delay(attempt, base_delay, max_delay, rng))
    assert last is not None
    raise last


class Resilient:
    """Bundle of breaker + retry + timeout around a flaky dependency call."""

    def __init__(self, name: str, *, attempts: int = 3, base_delay: float = 0.1, max_delay: float = 2.0,
                 timeout: float | None = 5.0, breaker: CircuitBreaker | None = None,
                 sleep: Callable[[float], None] = time.sleep, rng: Callable[[], float] = random.random):
        self.name = name
        self.attempts = attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.breaker = breaker or CircuitBreaker(name)
        self._sleep = sleep
        self._rng = rng
        self.calls = 0
        self.retries = 0

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        if not self.breaker.allow():
            raise CircuitOpenError(f"circuit '{self.name}' is open")
        self.calls += 1

        def _count(_a: int, _e: BaseException, _d: float) -> None:
            self.retries += 1

        try:
            result = retry(
                lambda: call_with_timeout(fn, self.timeout, *args, **kwargs),
                attempts=self.attempts, base_delay=self.base_delay, max_delay=self.max_delay,
                sleep=self._sleep, rng=self._rng, on_retry=_count,
            )
        except Exception:
            self.breaker.record_failure()
            raise
        self.breaker.record_success()
        return result

    def stats(self) -> dict[str, Any]:
        return {"name": self.name, "state": self.breaker.state, "calls": self.calls, "retries": self.retries}
