"""The searchable index: chunks + BM25 + vector store + per-role ACL views + persistence."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from .auth import can_access
from .bm25 import BM25Index
from .embeddings import Embedder
from .models import Chunk
from .text import tokens
from .vectorstore import InMemoryVectorStore, VectorStore


class Index:
    def __init__(self, embedder: Embedder, vectors: VectorStore | None = None):
        self.embedder = embedder
        self.vectors: VectorStore = vectors if vectors is not None else InMemoryVectorStore(embedder.dim)
        self.bm25 = BM25Index()
        self.chunks: dict[str, Chunk] = {}
        self.manifest: dict[str, dict] = {}  # doc_id -> {hash, chunk_ids, vectors}
        self.version = 0
        self._acl_cache: dict[tuple[int, frozenset[str]], frozenset[str]] = {}
        self._lock = threading.RLock()

    # ---- writes -------------------------------------------------------------------------------
    def upsert_document(self, doc_id: str, chunks: list[Chunk], vectors: np.ndarray | None, content_hash: str) -> None:
        with self._lock:
            self._remove_locked(doc_id)
            for c in chunks:
                self.chunks[c.chunk_id] = c
                self.bm25.add(c.chunk_id, tokens(c.index_text))
            if vectors is not None and len(chunks):
                self.vectors.upsert([c.chunk_id for c in chunks], vectors)
            self.manifest[doc_id] = {
                "hash": content_hash,
                "chunk_ids": [c.chunk_id for c in chunks],
                "vectors": vectors is not None,
            }
            self._bump()

    def remove_document(self, doc_id: str) -> None:
        with self._lock:
            self._remove_locked(doc_id)
            self._bump()

    def _remove_locked(self, doc_id: str) -> None:
        entry = self.manifest.pop(doc_id, None)
        if not entry:
            return
        for cid in entry["chunk_ids"]:
            self.chunks.pop(cid, None)
            self.bm25.remove(cid)
        self.vectors.delete(entry["chunk_ids"])

    def _bump(self) -> None:
        self.version += 1
        self._acl_cache.clear()

    # ---- reads --------------------------------------------------------------------------------
    def allowed_chunk_ids(self, effective_roles: frozenset[str]) -> frozenset[str]:
        """Chunk ids this role set may read. Cached per (index version, roles)."""
        key = (self.version, effective_roles)
        with self._lock:
            hit = self._acl_cache.get(key)
            if hit is None:
                hit = frozenset(cid for cid, c in self.chunks.items() if can_access(effective_roles, c.allowed_roles))
                self._acl_cache[key] = hit
            return hit

    def vocabulary(self):
        return self.bm25.vocabulary()

    def __len__(self) -> int:
        return len(self.chunks)

    def stats(self) -> dict:
        return {"documents": len(self.manifest), "chunks": len(self.chunks), "vectors": len(self.vectors),
                "version": self.version, "embedder": self.embedder.name}

    # ---- persistence --------------------------------------------------------------------------
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "chunks.json").write_text(json.dumps([c.to_dict() for c in self.chunks.values()]))
        (directory / "manifest.json").write_text(json.dumps(self.manifest, indent=1))
        self.vectors.save(directory / "vectors")

    def load(self, directory: Path) -> bool:
        if not (directory / "manifest.json").exists():
            return False
        chunks = [Chunk.from_dict(d) for d in json.loads((directory / "chunks.json").read_text())]
        with self._lock:
            self.chunks = {c.chunk_id: c for c in chunks}
            self.manifest = json.loads((directory / "manifest.json").read_text())
            self.bm25 = BM25Index()
            for c in chunks:
                self.bm25.add(c.chunk_id, tokens(c.index_text))
            if (directory / "vectors.npy").exists():
                self.vectors.load(directory / "vectors")
            self._bump()
        return True
