"""Runtime configuration. Everything is overridable via environment variables (see .env.example)."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _f(env: Mapping[str, str], key: str, default: float) -> float:
    return float(env.get(key, default))


def _i(env: Mapping[str, str], key: str, default: int) -> int:
    return int(env.get(key, default))


@dataclass(frozen=True)
class Settings:
    # paths
    corpus_dir: Path = ROOT / "data" / "corpus"
    db_path: Path = ROOT / ".data" / "guardrail.db"
    regression_path: Path = ROOT / "data" / "evals" / "regression.jsonl"
    golden_path: Path = ROOT / "data" / "evals" / "golden.jsonl"
    index_dir: Path | None = None  # when set the index is persisted/loaded here

    # providers
    embedder: str = "hashing"  # hashing | ollama
    llm: str = "mock"  # mock | anthropic | ollama
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5-5"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    ollama_embed_model: str = "nomic-embed-text"
    price_in_per_mtok: float = 3.0
    price_out_per_mtok: float = 15.0

    # ingestion
    chunk_size: int = 80  # words
    chunk_overlap: int = 20  # words

    # retrieval
    retrieval_mode: str = "hybrid"  # hybrid | bm25 | vector
    rerank: bool = True
    top_k: int = 5
    candidates: int = 20
    rrf_k: int = 60
    context_chunks: int = 4  # chunks passed to the LLM

    # guardrail / HITL thresholds
    min_evidence: float = 0.35
    max_unknown_fraction: float = 0.4  # share of query weight on terms absent from the whole corpus
    review_confidence: float = 0.55
    support_threshold: float = 0.7
    max_question_chars: int = 2000
    allowed_email_domains: tuple[str, ...] = ("acmedynamics.example",)

    # reliability
    retry_attempts: int = 3
    retry_base_delay_s: float = 0.1
    retry_max_delay_s: float = 2.0
    embed_timeout_s: float = 2.0
    llm_timeout_s: float = 30.0
    breaker_threshold: int = 5
    breaker_recovery_s: float = 30.0
    rate_limit_per_min: int = 120
    rate_limit_burst: int = 30
    max_concurrency: int = 32
    request_timeout_s: float = 20.0

    # auth / observability
    demo_keys: bool = True
    api_keys: dict = field(default_factory=dict)
    otlp_endpoint: str = ""

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        e = os.environ if env is None else env
        api_keys = json.loads(e["RAG_API_KEYS"]) if e.get("RAG_API_KEYS") else {}
        d = cls()
        index_dir = e.get("RAG_INDEX_DIR")
        return cls(
            corpus_dir=Path(e.get("RAG_CORPUS_DIR", d.corpus_dir)),
            db_path=Path(e.get("RAG_DB_PATH", d.db_path)),
            regression_path=Path(e.get("RAG_REGRESSION_PATH", d.regression_path)),
            golden_path=Path(e.get("RAG_GOLDEN_PATH", d.golden_path)),
            index_dir=Path(index_dir) if index_dir else None,
            embedder=e.get("RAG_EMBEDDER", d.embedder),
            llm=e.get("RAG_LLM", d.llm),
            anthropic_api_key=e.get("ANTHROPIC_API_KEY", ""),
            anthropic_model=e.get("ANTHROPIC_MODEL", d.anthropic_model),
            ollama_host=e.get("OLLAMA_HOST", d.ollama_host),
            ollama_model=e.get("OLLAMA_MODEL", d.ollama_model),
            ollama_embed_model=e.get("OLLAMA_EMBED_MODEL", d.ollama_embed_model),
            price_in_per_mtok=_f(e, "RAG_PRICE_IN_PER_MTOK", d.price_in_per_mtok),
            price_out_per_mtok=_f(e, "RAG_PRICE_OUT_PER_MTOK", d.price_out_per_mtok),
            chunk_size=_i(e, "RAG_CHUNK_SIZE", d.chunk_size),
            chunk_overlap=_i(e, "RAG_CHUNK_OVERLAP", d.chunk_overlap),
            retrieval_mode=e.get("RAG_RETRIEVAL_MODE", d.retrieval_mode),
            rerank=e.get("RAG_RERANK", "1") not in ("0", "false", "False"),
            top_k=_i(e, "RAG_TOP_K", d.top_k),
            candidates=_i(e, "RAG_CANDIDATES", d.candidates),
            rrf_k=_i(e, "RAG_RRF_K", d.rrf_k),
            context_chunks=_i(e, "RAG_CONTEXT_CHUNKS", d.context_chunks),
            min_evidence=_f(e, "RAG_MIN_EVIDENCE", d.min_evidence),
            max_unknown_fraction=_f(e, "RAG_MAX_UNKNOWN_FRACTION", d.max_unknown_fraction),
            review_confidence=_f(e, "RAG_REVIEW_CONFIDENCE", d.review_confidence),
            support_threshold=_f(e, "RAG_SUPPORT_THRESHOLD", d.support_threshold),
            max_question_chars=_i(e, "RAG_MAX_QUESTION_CHARS", d.max_question_chars),
            retry_attempts=_i(e, "RAG_RETRY_ATTEMPTS", d.retry_attempts),
            embed_timeout_s=_f(e, "RAG_EMBED_TIMEOUT_S", d.embed_timeout_s),
            llm_timeout_s=_f(e, "RAG_LLM_TIMEOUT_S", d.llm_timeout_s),
            breaker_threshold=_i(e, "RAG_BREAKER_THRESHOLD", d.breaker_threshold),
            breaker_recovery_s=_f(e, "RAG_BREAKER_RECOVERY_S", d.breaker_recovery_s),
            rate_limit_per_min=_i(e, "RAG_RATE_LIMIT_PER_MIN", d.rate_limit_per_min),
            rate_limit_burst=_i(e, "RAG_RATE_LIMIT_BURST", d.rate_limit_burst),
            max_concurrency=_i(e, "RAG_MAX_CONCURRENCY", d.max_concurrency),
            request_timeout_s=_f(e, "RAG_REQUEST_TIMEOUT_S", d.request_timeout_s),
            demo_keys=e.get("RAG_DEMO_KEYS", "1") not in ("0", "false", "False"),
            api_keys=api_keys,
            otlp_endpoint=e.get("OTEL_EXPORTER_OTLP_ENDPOINT", ""),
        )
