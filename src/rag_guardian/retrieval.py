"""Hybrid retrieval (BM25 + vector, reciprocal rank fusion), ACL pre-filtering and reranking."""

from __future__ import annotations

from dataclasses import dataclass, field

from .embeddings import Embedder, EmbedderUnavailable
from .index import Index
from .models import Hit
from .text import tokens


@dataclass
class RetrievalResult:
    hits: list[Hit]
    degraded: list[str] = field(default_factory=list)
    mode: str = "hybrid"


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """RRF: score(d) = sum over rankers of 1 / (k + rank_d). Rank is 1-based."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


class HybridRetriever:
    def __init__(self, index: Index, embedder: Embedder, mode: str = "hybrid", candidates: int = 20, rrf_k: int = 60):
        if mode not in ("hybrid", "bm25", "vector"):
            raise ValueError(f"unknown retrieval mode {mode!r}")
        self.index = index
        self.embedder = embedder
        self.mode = mode
        self.candidates = candidates
        self.rrf_k = rrf_k

    def retrieve(self, query: str, effective_roles: frozenset[str], k: int = 20) -> RetrievalResult:
        # ACL is applied *before* scoring: restricted chunks never enter either ranker.
        allowed = self.index.allowed_chunk_ids(effective_roles)
        degraded: list[str] = []
        bm_ranked: list[tuple[str, float]] = []
        vec_ranked: list[tuple[str, float]] = []
        want_vec = self.mode in ("hybrid", "vector")
        want_bm = self.mode in ("hybrid", "bm25")
        if want_vec:
            try:
                qv = self.embedder.embed([query])[0]
                vec_ranked = self.index.vectors.search(qv, self.candidates, allowed)
            except EmbedderUnavailable:
                degraded.append("embedder_unavailable:lexical_only")
                want_bm = True  # graceful degradation: fall back to lexical retrieval
        if want_bm:
            bm_ranked = self.index.bm25.search(tokens(query), self.candidates, allowed)
        bm_rank = {cid: r for r, (cid, _) in enumerate(bm_ranked, start=1)}
        vec_rank = {cid: r for r, (cid, _) in enumerate(vec_ranked, start=1)}
        fused = reciprocal_rank_fusion([[c for c, _ in bm_ranked], [c for c, _ in vec_ranked]], self.rrf_k)
        order = sorted(fused, key=lambda c: (-fused[c], c))[:k]
        hits = [Hit(self.index.chunks[c], bm25_rank=bm_rank.get(c), vector_rank=vec_rank.get(c), rrf=fused[c]) for c in order]
        return RetrievalResult(hits=hits, degraded=degraded, mode=self.mode)


class Reranker:
    """Deterministic cross-scorer: idf-weighted term coverage + bigram match + title/section match.

    ``coverage`` doubles as the evidence signal: if even the best chunk covers little of what the
    user asked about, the system refuses instead of improvising.
    """

    def __init__(self, index: Index):
        self.index = index

    def unknown_fraction(self, query: str) -> float:
        """Share of the query's idf weight carried by terms that appear nowhere in the corpus.

        A high value means the question is about something the corpus never mentions (e.g. 'sabbatical'),
        so a chunk that merely shares generic words ('leave', 'days') must not be treated as evidence.
        """
        terms = list(dict.fromkeys(tokens(query)))
        if not terms:
            return 1.0
        weight = {t: self.index.bm25.idf(t) for t in terms}
        total = sum(weight.values()) or 1.0
        return sum(w for t, w in weight.items() if self.index.bm25.df(t) == 0) / total

    def rerank(self, query: str, hits: list[Hit], top_k: int, reorder: bool = True) -> list[Hit]:
        qtoks = tokens(query)
        qterms = list(dict.fromkeys(qtoks))
        if not qterms or not hits:
            return hits[:top_k]
        weight = {t: self.index.bm25.idf(t) for t in qterms}
        total = sum(weight.values()) or 1.0
        qbigrams = set(zip(qtoks, qtoks[1:], strict=False))
        max_rrf = max(h.rrf for h in hits) or 1.0
        for h in hits:
            ctoks = tokens(h.chunk.index_text)
            cset = set(ctoks)
            h.coverage = sum(weight[t] for t in qterms if t in cset) / total
            cbigrams = set(zip(ctoks, ctoks[1:], strict=False))
            bigram = len(qbigrams & cbigrams) / len(qbigrams) if qbigrams else 0.0
            head = set(tokens(f"{h.chunk.title} {h.chunk.section}"))
            title = sum(weight[t] for t in qterms if t in head) / total
            h.rerank = 0.65 * h.coverage + 0.20 * bigram + 0.10 * title + 0.05 * (h.rrf / max_rrf)
        if not reorder:  # ablation: keep fusion order but still expose coverage as the evidence signal
            return hits[:top_k]
        return sorted(hits, key=lambda h: (-h.rerank, h.chunk.chunk_id))[:top_k]
