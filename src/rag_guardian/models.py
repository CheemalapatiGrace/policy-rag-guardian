"""Core data types shared across the package."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str  # clean text shown to the LLM / in citations
    index_text: str  # title + section + text, used for lexical + vector indexing
    position: int
    allowed_roles: tuple[str, ...]
    classification: str = "internal"
    sensitive: bool = False
    source: str = ""
    content_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["allowed_roles"] = list(self.allowed_roles)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chunk:
        d = dict(d)
        d["allowed_roles"] = tuple(d["allowed_roles"])
        return cls(**d)


@dataclass
class Hit:
    chunk: Chunk
    bm25_rank: int | None = None
    vector_rank: int | None = None
    rrf: float = 0.0
    coverage: float = 0.0  # idf-weighted share of query terms present in the chunk (0..1)
    rerank: float = 0.0  # final reranker score (0..1)

    def summary(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "bm25_rank": self.bm25_rank,
            "vector_rank": self.vector_rank,
            "rrf": round(self.rrf, 5),
            "coverage": round(self.coverage, 3),
            "rerank": round(self.rerank, 3),
        }


@dataclass
class Citation:
    marker: int
    chunk_id: str
    doc_id: str
    title: str
    section: str
    snippet: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QueryResult:
    trace_id: str
    status: str  # answered | refused | blocked | pending_review
    message: str  # always safe to show the asker (answer, refusal text or hold notice)
    answer: str | None = None  # only set when released
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    review_id: int | None = None
    retrieved: list[dict[str, Any]] = field(default_factory=list)  # internal (evals / traces)
    faithfulness: float | None = None

    def public(self) -> dict[str, Any]:
        # a held answer must not leak through its citation snippets before a human releases it
        held = self.status == "pending_review"
        return {
            "trace_id": self.trace_id,
            "status": self.status,
            "message": self.message,
            "answer": self.answer,
            "citations": [] if held else [c.to_dict() for c in self.citations],
            "confidence": round(self.confidence, 3),
            "reasons": self.reasons,
            "degraded": self.degraded,
            "latency_ms": round(self.latency_ms, 2),
            "review_id": self.review_id,
        }
