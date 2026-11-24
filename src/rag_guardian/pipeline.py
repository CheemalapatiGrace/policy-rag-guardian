"""End-to-end orchestration: guardrail_in -> retrieve -> rerank -> generate -> guardrail_out -> release/hold."""

from __future__ import annotations

import asyncio
import re
from typing import Any

from .auth import Principal
from .config import Settings
from .embeddings import CachedEmbedder, HashingEmbedder, OllamaEmbedder, ResilientEmbedder
from .guardrails import ContextGuard, InputGuard, OutputGuard
from .hitl import ReviewService
from .index import Index
from .ingestion import Ingestor, IngestReport
from .llm import LLM, AnthropicLLM, MockLLM, OllamaLLM, ResilientLLM, estimate_cost
from .models import Citation, Hit, QueryResult
from .resilience import CircuitBreaker, Resilient
from .retrieval import HybridRetriever, Reranker
from .storage import Database
from .tracing import OtlpExporter, Trace

# a redaction token and the (up to two) words introducing it ("my SSN is [REDACTED_SSN]") carry no retrieval signal
_PLACEHOLDER = re.compile(r"(?:\b\w+\b[\s:,]+){0,2}\[REDACTED_[A-Z_]+\]")
MSG_BLOCKED = "Your request was blocked by security guardrails (possible prompt injection)."
MSG_SCOPE = "I can only answer questions about company policies."
MSG_NO_EVIDENCE = "I couldn't find this in the policy documents available to you."
MSG_HELD = "Your question was routed to a human reviewer before release. Check back with this reference."
MSG_ERROR = "Something went wrong; the request was refused safely. Please try again."


class RAGPipeline:
    def __init__(self, settings: Settings, index: Index, retriever: HybridRetriever, reranker: Reranker, llm: LLM,
                 input_guard: InputGuard, context_guard: ContextGuard, output_guard: OutputGuard, db: Database,
                 reviews: ReviewService, exporter: OtlpExporter | None = None, embedder: Any = None,
                 policies: list[Resilient] | None = None):
        self.settings = settings
        self.index = index
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.input_guard = input_guard
        self.context_guard = context_guard
        self.output_guard = output_guard
        self.db = db
        self.reviews = reviews
        self.exporter = exporter
        self.embedder = embedder
        self.policies = policies or []

    # ---- public API ---------------------------------------------------------------------------
    def answer(self, question: str, principal: Principal, top_k: int | None = None) -> QueryResult:
        trace = Trace(principal.user_id, principal.label)
        try:
            result = self._run(trace, question, principal, top_k or self.settings.top_k)
        except Exception as exc:  # noqa: BLE001 - fail closed: never release anything on an internal error
            trace.status, trace.reason = "refused", f"internal_error:{type(exc).__name__}"
            trace.final_answer = None
            result = QueryResult(trace.trace_id, "refused", MSG_ERROR, reasons=[trace.reason])
        trace.finish()
        result.latency_ms = trace.latency_ms
        data = trace.to_dict()
        self.db.save_trace(data)
        self.db.append_audit(principal.user_id, principal.label, "query", f"trace:{trace.trace_id}",
                             {"status": result.status, "reasons": result.reasons})
        if self.exporter:
            self.exporter.export(data)
        return result

    async def aanswer(self, question: str, principal: Principal, top_k: int | None = None) -> QueryResult:
        return await asyncio.to_thread(self.answer, question, principal, top_k)

    def reingest(self) -> IngestReport:
        """Idempotent re-ingest of the corpus directory (unchanged docs are skipped by content hash)."""
        report = Ingestor(self.index, self.embedder, self.settings.chunk_size,
                          self.settings.chunk_overlap).ingest_directory(self.settings.corpus_dir)
        if self.settings.index_dir:
            self.index.save(self.settings.index_dir)
        return report

    def health(self) -> dict[str, Any]:
        breakers = {p.name: p.breaker.state for p in self.policies}
        cache = self.embedder.stats() if hasattr(self.embedder, "stats") else {}
        return {"index": self.index.stats(), "breakers": breakers, "embedding_cache": cache,
                "llm": self.llm.name, "retrieval_mode": self.retriever.mode}

    # ---- internals ----------------------------------------------------------------------------
    def _conclude(self, trace: Trace, status: str, message: str, reasons: list[str], *, answer: str | None = None,
                  citations: list[Citation] | None = None, confidence: float = 0.0, review_id: int | None = None,
                  retrieved: list[dict] | None = None, faithfulness: float | None = None) -> QueryResult:
        trace.status = status
        trace.reason = ",".join(reasons)
        trace.confidence = confidence
        trace.final_answer = answer
        trace.citations = [c.to_dict() for c in citations or []]
        return QueryResult(trace.trace_id, status, message, answer, citations or [], confidence, reasons,
                           list(trace.degraded), 0.0, review_id, retrieved or [], faithfulness)

    def _run(self, trace: Trace, question: str, principal: Principal, k: int) -> QueryResult:
        s = self.settings
        roles = principal.effective

        # 1. input guardrail -------------------------------------------------------------------
        with trace.span("guardrail_in") as sp:
            verdict = self.input_guard.check(question)
            sp.input = {"question": verdict.redacted}
            sp.output = {key: val for key, val in verdict.to_dict().items() if key != "redacted"}
            if not verdict.allowed:
                sp.status = "blocked"
        trace.question = verdict.redacted
        if not verdict.allowed:
            if verdict.reason == "prompt_injection":
                return self._conclude(trace, "blocked", MSG_BLOCKED, ["prompt_injection"])
            msg = MSG_SCOPE if verdict.reason == "out_of_scope" else f"Invalid request: {verdict.reason}."
            return self._conclude(trace, "refused", msg, [verdict.reason or "rejected"])
        query = verdict.redacted
        search_query = _PLACEHOLDER.sub(" ", query)  # redaction tokens carry no retrieval signal

        # 2. retrieve (ACL pre-filter + hybrid) + retrieval-side guard --------------------------
        with trace.span("retrieve") as sp:
            sp.input = {"query": query, "roles": sorted(roles), "candidates": s.candidates, "mode": self.retriever.mode}
            rr = self.retriever.retrieve(search_query, roles, k=s.candidates)
            safe, dropped = self.context_guard.filter([h.chunk for h in rr.hits], roles)
            keep = {c.chunk_id for c in safe}
            hits = [h for h in rr.hits if h.chunk.chunk_id in keep]
            sp.output = {"hits": [h.summary() for h in hits[:10]], "quarantined": dropped}
            if rr.degraded:
                sp.status = "degraded"
                sp.attrs["degraded"] = rr.degraded
                trace.degraded += rr.degraded

        # 3. rerank + evidence check -----------------------------------------------------------
        with trace.span("rerank") as sp:
            sp.input = {"candidates": len(hits), "enabled": s.rerank}
            top: list[Hit] = self.reranker.rerank(search_query, hits, k, reorder=s.rerank)
            unknown = self.reranker.unknown_fraction(search_query)
            sp.output = {"top": [h.summary() for h in top], "unknown_term_fraction": round(unknown, 3)}
        retrieved = [{"doc_id": h.chunk.doc_id, "chunk_id": h.chunk.chunk_id, "rerank": round(h.rerank, 3)} for h in top]
        if not top or top[0].coverage < s.min_evidence or unknown > s.max_unknown_fraction:
            return self._conclude(trace, "refused", MSG_NO_EVIDENCE, ["no_evidence"], retrieved=retrieved)

        # 4. generate ---------------------------------------------------------------------------
        ctx = [h.chunk for h in top[: s.context_chunks]]
        with trace.span("generate") as sp:
            sp.input = {"question": query, "context_chunk_ids": [c.chunk_id for c in ctx], "model": self.llm.name}
            gen = self.llm.generate(query, ctx)
            cost = estimate_cost(gen, s.price_in_per_mtok, s.price_out_per_mtok)
            trace.tokens_in += gen.input_tokens
            trace.tokens_out += gen.output_tokens
            trace.cost_usd += cost
            sp.output = {"text": gen.text}
            sp.attrs.update({"model": gen.model, "tokens_in": gen.input_tokens, "tokens_out": gen.output_tokens,
                             "cost_usd": round(cost, 6)})
            if gen.degraded:
                sp.status = "degraded"
                trace.degraded.append("llm_unavailable:extractive_fallback")

        # 5. output guardrail ------------------------------------------------------------------
        with trace.span("guardrail_out") as sp:
            ov = self.output_guard.check(gen.text, ctx, roles)
            sp.input = {"answer": gen.text, "cited_chunks": [c.chunk_id for c in ctx]}
            sp.output = ov.to_dict()
            if not ov.ok:
                sp.status = "blocked"
        if ov.refusal:
            return self._conclude(trace, "refused", MSG_NO_EVIDENCE, ["no_evidence"], retrieved=retrieved)
        if "acl_violation" in ov.issues:
            return self._conclude(trace, "refused", MSG_NO_EVIDENCE, ["acl_violation"], retrieved=retrieved)

        used = sorted({m for cl in ov.claims for m in cl.markers if 1 <= m <= len(ctx)})
        citations = [Citation(m, ctx[m - 1].chunk_id, ctx[m - 1].doc_id, ctx[m - 1].title, ctx[m - 1].section,
                              ctx[m - 1].text[:220]) for m in used]
        confidence = round(0.6 * top[0].coverage + 0.4 * ov.faithfulness, 4)

        # 6. release or hold for human review ---------------------------------------------------
        reasons: list[str] = []
        if {"unsupported_claims", "invalid_citations", "no_claims"} & set(ov.issues):
            reasons.append("failed_grounding")
        if "pii_leak" in ov.issues:
            reasons.append("pii_in_output")
        if confidence < s.review_confidence:
            reasons.append("low_confidence")
        if verdict.sensitive or any(ctx[m - 1].sensitive for m in used):
            reasons.append("sensitive_topic")
        if reasons:
            draft = ov.clean_text
            rid = self.reviews.enqueue(trace.trace_id, reasons, query, principal, draft, [c.to_dict() for c in citations])
            res = self._conclude(trace, "pending_review", f"{MSG_HELD} (ref {trace.trace_id[:8]})", reasons,
                                 citations=citations, confidence=confidence, review_id=rid, retrieved=retrieved,
                                 faithfulness=ov.faithfulness)
            trace.final_answer = None  # the draft lives in the review record until a human releases it
            return res
        return self._conclude(trace, "answered", ov.clean_text, ["grounded"], answer=ov.clean_text, citations=citations,
                              confidence=confidence, retrieved=retrieved, faithfulness=ov.faithfulness)


def build_pipeline(settings: Settings | None = None, *, db: Database | None = None, llm: LLM | None = None,
                   base_embedder: Any = None, ingest: bool = True) -> RAGPipeline:
    """Wire the default offline stack (override pieces for tests / real providers)."""
    s = settings or Settings.from_env()

    def policy(name: str, timeout: float) -> Resilient:
        return Resilient(name, attempts=s.retry_attempts, base_delay=s.retry_base_delay_s, max_delay=s.retry_max_delay_s,
                         timeout=timeout, breaker=CircuitBreaker(name, s.breaker_threshold, s.breaker_recovery_s))

    embed_policy = policy("embedder", s.embed_timeout_s)
    llm_policy = policy("llm", s.llm_timeout_s)

    if base_embedder is None:
        base_embedder = OllamaEmbedder(s.ollama_host, s.ollama_embed_model) if s.embedder == "ollama" else HashingEmbedder()
    embedder = CachedEmbedder(ResilientEmbedder(base_embedder, embed_policy))

    index = Index(embedder)
    if s.index_dir:
        index.load(s.index_dir)
    if ingest:
        Ingestor(index, embedder, s.chunk_size, s.chunk_overlap).ingest_directory(s.corpus_dir)
        if s.index_dir:
            index.save(s.index_dir)

    if llm is None:
        if s.llm == "anthropic":
            llm = ResilientLLM(AnthropicLLM(s.anthropic_api_key, s.anthropic_model), llm_policy)
        elif s.llm == "ollama":
            llm = ResilientLLM(OllamaLLM(s.ollama_host, s.ollama_model), llm_policy)
        else:
            llm = MockLLM()

    db = db or Database(s.db_path)
    exporter = OtlpExporter(s.otlp_endpoint) if s.otlp_endpoint else None
    return RAGPipeline(
        s, index, HybridRetriever(index, embedder, s.retrieval_mode, s.candidates, s.rrf_k), Reranker(index), llm,
        InputGuard(index.vocabulary, s.max_question_chars, scope_min_known=0.25, allowed_email_domains=s.allowed_email_domains),
        ContextGuard(), OutputGuard(s.support_threshold, s.allowed_email_domains), db,
        ReviewService(db, s.regression_path, s.allowed_email_domains), exporter, embedder, [embed_policy, llm_policy])


__all__ = ["RAGPipeline", "build_pipeline", "IngestReport"]
