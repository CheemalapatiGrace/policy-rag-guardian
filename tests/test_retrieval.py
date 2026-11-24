import numpy as np

from conftest import EMPLOYEE, GUEST, HR
from rag_guardian.bm25 import BM25Index
from rag_guardian.embeddings import CachedEmbedder, FlakyEmbedder, HashingEmbedder, ResilientEmbedder
from rag_guardian.pipeline import build_pipeline
from rag_guardian.resilience import CircuitBreaker, Resilient
from rag_guardian.retrieval import reciprocal_rank_fusion
from rag_guardian.storage import Database
from rag_guardian.vectorstore import InMemoryVectorStore


def test_bm25_ranks_by_term_relevance_and_supports_removal():
    bm = BM25Index()
    bm.add("a", ["leave", "leave", "leave", "annual"])
    bm.add("b", ["expense", "report"])
    bm.add("c", ["leave", "other", "other", "other"])
    ranked = [d for d, _ in bm.search(["leave"], 5)]
    assert ranked[0] == "a" and "b" not in ranked
    bm.remove("a")
    assert [d for d, _ in bm.search(["leave"], 5)] == ["c"] and len(bm) == 2


def test_bm25_acl_prefilter_hides_forbidden_docs():
    bm = BM25Index()
    bm.add("secret", ["salary", "salary"])
    bm.add("open", ["salary"])
    assert [d for d, _ in bm.search(["salary"], 5, allowed={"open"})] == ["open"]


def test_rrf_rewards_agreement_between_rankers():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "b", "d"]], k=60)
    assert fused["b"] > fused["a"] and fused["b"] > fused["d"]
    assert abs(fused["a"] - 1 / 61) < 1e-9


def test_vector_store_upsert_delete_and_filtered_search():
    vs = InMemoryVectorStore(3)
    vs.upsert(["x", "y", "z"], np.eye(3, dtype=np.float32))
    vs.upsert(["x"], np.array([[0.0, 1.0, 0.0]], dtype=np.float32))  # overwrite, not append
    assert len(vs) == 3
    q = np.array([0, 1, 0], dtype=np.float32)
    assert {i for i, s in vs.search(q, 2) if s > 0.99} == {"x", "y"}
    assert [i for i, _ in vs.search(q, 3, allowed={"z"})] == ["z"]
    vs.delete(["x"])
    assert len(vs) == 2 and "x" not in [i for i, _ in vs.search(q, 3)]


def test_custom_vector_store_can_be_plugged_in(settings):
    class Recording(InMemoryVectorStore):
        searches = 0

        def search(self, vector, k, allowed=None):
            Recording.searches += 1
            return super().search(vector, k, allowed)

    from rag_guardian.index import Index
    from rag_guardian.ingestion import Ingestor
    from rag_guardian.retrieval import HybridRetriever

    emb = CachedEmbedder(HashingEmbedder())
    idx = Index(emb, Recording(emb.dim))
    Ingestor(idx, emb).ingest_directory(settings.corpus_dir)
    HybridRetriever(idx, emb).retrieve("annual leave days", EMPLOYEE.effective, 5)
    assert Recording.searches == 1


def test_hybrid_retrieval_finds_relevant_doc(pipeline):
    res = pipeline.retriever.retrieve("how many vacation days per year", EMPLOYEE.effective, 5)
    assert res.hits[0].chunk.doc_id == "hr-leave-policy"
    assert res.hits[0].bm25_rank and res.hits[0].vector_rank  # both rankers contributed


def test_retrieval_never_returns_documents_above_clearance(pipeline):
    for q in ["salary band level 4", "break-glass account", "procurement committee quotes", "annual leave"]:
        for who in (GUEST, EMPLOYEE):
            docs = {h.chunk.doc_id for h in pipeline.retriever.retrieve(q, who.effective, 20).hits}
            assert not docs & {"compensation-bands", "incident-response-runbook", "vendor-procurement-policy"}
    hr_docs = {h.chunk.doc_id for h in pipeline.retriever.retrieve("salary band level 4", HR.effective, 5).hits}
    assert "compensation-bands" in hr_docs


def test_vector_only_and_bm25_only_modes(settings):
    from dataclasses import replace

    for mode in ("bm25", "vector"):
        p = build_pipeline(replace(settings, retrieval_mode=mode), db=Database(":memory:"))
        top = p.retriever.retrieve("legal hold retention", EMPLOYEE.effective, 3).hits[0]
        assert top.chunk.doc_id == "data-retention-policy"
        assert (top.bm25_rank is None) == (mode == "vector")


def test_embedder_outage_falls_back_to_lexical_only(settings):
    flaky = FlakyEmbedder(HashingEmbedder())
    p = build_pipeline(settings, db=Database(":memory:"), base_embedder=flaky)
    flaky.always_fail = True
    res = p.retriever.retrieve("how long are security logs retained", EMPLOYEE.effective, 5)
    assert res.degraded == ["embedder_unavailable:lexical_only"]
    assert res.hits and res.hits[0].chunk.doc_id == "data-retention-policy" and res.hits[0].vector_rank is None


def test_embedding_cache_avoids_recomputation():
    flaky = FlakyEmbedder(HashingEmbedder())
    cache = CachedEmbedder(ResilientEmbedder(flaky, Resilient("e", attempts=1, breaker=CircuitBreaker())))
    a = cache.embed(["annual leave days", "annual leave days", "carry over"])
    calls = flaky.calls
    b = cache.embed(["annual leave days", "carry over"])
    assert flaky.calls == calls and np.allclose(a[0], b[0]) and np.linalg.norm(a[0]) > 0 and cache.stats()["hit_rate"] >= 0.4


def test_reranker_scores_coverage_and_unknown_terms(pipeline):
    rr = pipeline.reranker
    assert rr.unknown_fraction("sabbatical leave") > 0.4 > rr.unknown_fraction("annual leave")
    hits = pipeline.retriever.retrieve("hotel cap per night", EMPLOYEE.effective, 10).hits
    top = rr.rerank("hotel cap per night", hits, 3)
    assert top[0].chunk.doc_id == "expense-policy" and top[0].coverage > 0.9
