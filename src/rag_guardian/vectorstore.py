"""Vector store interface + in-memory NumPy implementation (brute-force cosine, ACL pre-filter)."""

from __future__ import annotations

import json
from collections.abc import Collection, Iterable, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np


class VectorStore(Protocol):
    """Anything implementing this can replace the default store (FAISS, pgvector, Qdrant, ...)."""

    def upsert(self, ids: Sequence[str], vectors: np.ndarray) -> None: ...

    def delete(self, ids: Iterable[str]) -> None: ...

    def search(self, vector: np.ndarray, k: int, allowed: Collection[str] | None = None) -> list[tuple[str, float]]: ...

    def __len__(self) -> int: ...

    def save(self, path: Path) -> None: ...

    def load(self, path: Path) -> None: ...


class InMemoryVectorStore:
    """Row-matrix of L2-normalised float32 vectors; search is one matmul. Fine up to ~1e5 chunks."""

    def __init__(self, dim: int):
        self.dim = dim
        self._ids: list[str] = []
        self._row: dict[str, int] = {}
        self._mat = np.zeros((0, dim), dtype=np.float32)

    def __len__(self) -> int:
        return len(self._ids)

    def upsert(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        if len(ids) != len(vectors):
            raise ValueError("ids/vectors length mismatch")
        new_rows, new_ids = [], []
        for i, vid in enumerate(ids):
            if vid in self._row:
                self._mat[self._row[vid]] = vectors[i]
            else:
                new_rows.append(vectors[i])
                new_ids.append(vid)
        if new_rows:
            start = len(self._ids)
            self._mat = np.vstack([self._mat, np.asarray(new_rows, dtype=np.float32)]) if len(self._ids) else np.asarray(new_rows, dtype=np.float32)
            self._ids.extend(new_ids)
            for off, vid in enumerate(new_ids):
                self._row[vid] = start + off

    def delete(self, ids: Iterable[str]) -> None:
        drop = {i for i in ids if i in self._row}
        if not drop:
            return
        keep = [i for i, vid in enumerate(self._ids) if vid not in drop]
        self._mat = self._mat[keep] if keep else np.zeros((0, self.dim), dtype=np.float32)
        self._ids = [self._ids[i] for i in keep]
        self._row = {vid: i for i, vid in enumerate(self._ids)}

    def search(self, vector: np.ndarray, k: int, allowed: Collection[str] | None = None) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        scores = self._mat @ np.asarray(vector, dtype=np.float32)
        if allowed is not None:  # pre-filter: forbidden rows can never enter the ranking
            mask = np.fromiter((vid in allowed for vid in self._ids), dtype=bool, count=len(self._ids))
            if not mask.any():
                return []
            scores = np.where(mask, scores, -np.inf)
        k = min(k, len(self._ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self._ids[i], float(scores[i])) for i in top if np.isfinite(scores[i])]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path.with_suffix(".npy"), self._mat)
        path.with_suffix(".ids.json").write_text(json.dumps(self._ids))

    def load(self, path: Path) -> None:
        self._mat = np.load(path.with_suffix(".npy"))
        self._ids = json.loads(path.with_suffix(".ids.json").read_text())
        self._row = {vid: i for i, vid in enumerate(self._ids)}
        self.dim = self._mat.shape[1] if self._mat.size else self.dim
