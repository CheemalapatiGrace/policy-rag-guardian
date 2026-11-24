"""Embedding providers: deterministic hashing embedder (default), Ollama, plus cache and resilience."""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Protocol

import httpx
import numpy as np

from .resilience import CircuitOpenError, Resilient
from .text import tokens


class EmbedderUnavailable(RuntimeError):
    """The embedding backend could not be reached (after retries / breaker). Triggers lexical fallback."""


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Signed feature hashing over stemmed unigrams + bigrams, sublinear TF, L2-normalised.

    Fully deterministic (blake2b, not Python's randomised ``hash``) and needs no model download,
    so tests and CI are reproducible offline.
    """

    name = "hashing-v1"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")
        return h % self.dim, 1.0 if (h >> 63) & 1 else -1.0

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = tokens(text)
            feats: dict[str, float] = {}
            for t in toks:
                feats[t] = feats.get(t, 0.0) + 1.0
            for a, b in zip(toks, toks[1:], strict=False):
                feats[f"{a}_{b}"] = feats.get(f"{a}_{b}", 0.0) + 0.7
            for feat, tf in feats.items():
                idx, sign = self._bucket(feat)
                out[i, idx] += sign * (1.0 + np.log(tf))
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out


class OllamaEmbedder:
    """Embeddings from a local Ollama server (``/api/embeddings``). Optional, needs a running server."""

    def __init__(self, host: str, model: str, dim: int = 768, client: httpx.Client | None = None):
        self.name = f"ollama:{model}"
        self.dim = dim
        self._host = host.rstrip("/")
        self._model = model
        self._client = client or httpx.Client(timeout=30.0)

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            r = self._client.post(f"{self._host}/api/embeddings", json={"model": self._model, "prompt": t})
            r.raise_for_status()
            v = np.asarray(r.json()["embedding"], dtype=np.float32)
            n = np.linalg.norm(v)
            vecs.append(v / n if n else v)
        arr = np.vstack(vecs) if vecs else np.zeros((0, self.dim), dtype=np.float32)
        self.dim = arr.shape[1] if len(arr) else self.dim
        return arr


class ResilientEmbedder:
    """Retries + timeout + circuit breaker; raises :class:`EmbedderUnavailable` when exhausted."""

    def __init__(self, inner: Embedder, policy: Resilient):
        self.inner = inner
        self.policy = policy
        self.name = inner.name
        self.dim = inner.dim

    def embed(self, texts: list[str]) -> np.ndarray:
        try:
            return self.policy.call(self.inner.embed, texts)
        except CircuitOpenError as exc:
            raise EmbedderUnavailable(str(exc)) from exc
        except Exception as exc:
            raise EmbedderUnavailable(f"{type(exc).__name__}: {exc}") from exc


class CachedEmbedder:
    """Thread-safe LRU cache keyed by sha256(model + text). Hits never touch the backend."""

    def __init__(self, inner: Embedder, max_items: int = 50_000):
        self.inner = inner
        self.name = inner.name
        self.dim = inner.dim
        self._max = max_items
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.inner.name}\x00{text}".encode()).hexdigest()

    def embed(self, texts: list[str]) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        result: list[np.ndarray | None] = [None] * len(texts)
        missing: list[int] = []
        with self._lock:
            for i, k in enumerate(keys):
                if k in self._cache:
                    self._cache.move_to_end(k)
                    result[i] = self._cache[k]
                    self.hits += 1
                else:
                    missing.append(i)
                    self.misses += 1
        if missing:
            fresh = self.inner.embed([texts[i] for i in missing])
            self.dim = fresh.shape[1] if len(fresh) else self.dim
            with self._lock:
                for j, i in enumerate(missing):
                    result[i] = fresh[j]
                    self._cache[keys[i]] = fresh[j]
                while len(self._cache) > self._max:
                    self._cache.popitem(last=False)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack(result)  # type: ignore[arg-type]

    def stats(self) -> dict[str, float]:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses, "size": len(self._cache),
                "hit_rate": round(self.hits / total, 3) if total else 0.0}


class FlakyEmbedder:
    """Chaos helper: fails the first ``fail_first`` calls (or always, if ``always_fail``)."""

    def __init__(self, inner: Embedder, fail_first: int = 0, always_fail: bool = False):
        self.inner = inner
        self.name = inner.name
        self.dim = inner.dim
        self.fail_first = fail_first
        self.always_fail = always_fail
        self.calls = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        if self.always_fail or self.calls <= self.fail_first:
            raise ConnectionError("simulated embedder outage")
        return self.inner.embed(texts)
