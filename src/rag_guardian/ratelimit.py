"""Per-key token-bucket rate limiter (in-process; swap for Redis when running multiple replicas)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class TokenBucketLimiter:
    def __init__(self, per_minute: int = 120, burst: int = 30, clock: Callable[[], float] = time.monotonic):
        self.rate = per_minute / 60.0
        self.burst = float(burst)
        self._clock = clock
        self._lock = threading.Lock()
        self._state: dict[str, tuple[float, float]] = {}  # key -> (tokens, last_ts)

    def check(self, key: str) -> tuple[bool, float]:
        """Return (allowed, retry_after_seconds)."""
        with self._lock:
            now = self._clock()
            tokens, last = self._state.get(key, (self.burst, now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            if tokens >= 1.0:
                self._state[key] = (tokens - 1.0, now)
                return True, 0.0
            self._state[key] = (tokens, now)
            return False, (1.0 - tokens) / self.rate if self.rate else 60.0
