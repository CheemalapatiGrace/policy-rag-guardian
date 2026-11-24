from dataclasses import replace
from pathlib import Path

from conftest import EMPLOYEE  # noqa: F401
from rag_guardian.embeddings import CachedEmbedder, FlakyEmbedder, HashingEmbedder, ResilientEmbedder
from rag_guardian.index import Index
from rag_guardian.ingestion import Ingestor, chunk_document, load_documents, pack_sentences, parse_document
from rag_guardian.resilience import CircuitBreaker, Resilient

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"
DOC = "---\ndoc_id: d1\ntitle: Doc One\nallowed_roles: [hr]\nsensitive_sections: [Secret]\n---\n# Public\nAlpha beta gamma delta.\n# Secret\nHidden facts live here.\n"


def make(embedder=None):
    emb = embedder or CachedEmbedder(HashingEmbedder())
    idx = Index(emb)
    return idx, Ingestor(idx, emb, chunk_size=40, chunk_overlap=10)


def test_front_matter_is_parsed_into_acl_metadata():
    d = parse_document(DOC, "x/d1.md")
    assert (d.doc_id, d.title, d.allowed_roles, d.sensitive_sections) == ("d1", "Doc One", ("hr",), ("Secret",))
    chunks = chunk_document(d)
    assert {c.section: c.sensitive for c in chunks} == {"Public": False, "Secret": True}
    assert all(c.allowed_roles == ("hr",) and c.content_hash == d.content_hash for c in chunks)


def test_chunks_overlap_by_whole_sentences():
    sents = [f"alpha{i} " + "w " * 8 + "end." for i in range(12)]  # 10 words each
    groups = pack_sentences(sents, size=30, overlap=10)
    assert len(groups) > 2
    for a, b in zip(groups, groups[1:], strict=False):
        assert b[0] == a[-1]  # one-sentence overlap carried forward
    assert [s for g in groups for s in g][0] == sents[0]


def test_corpus_loads_with_expected_shape():
    docs = load_documents(CORPUS)
    assert len(docs) >= 8
    assert {d.doc_id for d in docs} >= {"hr-leave-policy", "compensation-bands", "incident-response-runbook"}


def test_reingest_is_idempotent_by_content_hash():
    idx, ing = make()
    first = ing.ingest_directory(CORPUS)
    version = idx.version
    second = ing.ingest_directory(CORPUS)
    assert len(first.added) == len(load_documents(CORPUS)) and not first.unchanged
    assert not second.added and not second.updated and len(second.unchanged) == len(first.added)
    assert idx.version == version  # nothing was touched


def test_changed_document_replaces_old_chunks_and_prunes_deleted():
    idx, ing = make()
    docs = [parse_document(DOC, "d1.md"), parse_document(DOC.replace("d1", "d2"), "d2.md")]
    ing.ingest_documents(docs)
    before = {c for c in idx.chunks if c.startswith("d1")}
    changed = parse_document(DOC.replace("Alpha beta gamma delta.", "Totally new text."), "d1.md")
    rep = ing.ingest_documents([changed])  # d2 absent -> pruned
    assert rep.updated == ["d1"] and rep.removed == ["d2"]
    assert not any(c.startswith("d2") for c in idx.chunks)
    assert before == {c for c in idx.chunks if c.startswith("d1")} and "Totally new" in idx.chunks["d1#000"].text


def test_embedder_outage_degrades_to_lexical_then_heals():
    flaky = FlakyEmbedder(HashingEmbedder(), always_fail=True)
    policy = Resilient("emb", attempts=1, timeout=1, breaker=CircuitBreaker(failure_threshold=100), sleep=lambda s: None)
    emb = CachedEmbedder(ResilientEmbedder(flaky, policy))
    idx, ing = make(emb)
    rep = ing.ingest_documents([parse_document(DOC, "d1.md")])
    assert rep.lexical_only == ["d1"] and idx.manifest["d1"]["vectors"] is False
    assert idx.bm25.search(["alpha"], 3)  # still searchable lexically
    flaky.always_fail = False
    rep2 = ing.ingest_documents([parse_document(DOC, "d1.md")])
    assert rep2.updated == ["d1"] and idx.manifest["d1"]["vectors"] is True and len(idx.vectors) == len(idx.chunks)


def test_index_persistence_roundtrip(tmp_path):
    idx, ing = make()
    ing.ingest_directory(CORPUS)
    idx.save(tmp_path / "ix")
    idx2 = Index(CachedEmbedder(HashingEmbedder()))
    assert idx2.load(tmp_path / "ix")
    assert idx2.stats()["chunks"] == idx.stats()["chunks"] and len(idx2.vectors) == len(idx.vectors)
    assert replace  # keep import used
