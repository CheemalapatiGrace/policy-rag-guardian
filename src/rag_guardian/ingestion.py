"""Ingestion pipeline: load -> parse front matter -> chunk (with overlap) -> embed -> index.

Idempotent by content hash: re-ingesting an unchanged corpus is a no-op, a changed document is
replaced atomically (old chunks removed), a deleted file is pruned. If the embedder is down, chunks
are still indexed lexically and flagged ``vectors=False`` so the next ingest retries the embedding.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .embeddings import Embedder, EmbedderUnavailable
from .index import Index
from .models import Chunk
from .text import clean_markdown, split_sentences

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


@dataclass
class Document:
    doc_id: str
    title: str
    body: str
    allowed_roles: tuple[str, ...]
    classification: str
    sensitive: bool
    source: str
    content_hash: str
    sensitive_sections: tuple[str, ...] = ()


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    lexical_only: list[str] = field(default_factory=list)
    chunks: int = 0
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {k: (v if not isinstance(v, list) else v) for k, v in self.__dict__.items()}


def parse_document(raw: str, source: str = "") -> Document:
    meta: dict = {}
    body = raw
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) == 3:
            meta = yaml.safe_load(parts[1]) or {}
            body = parts[2]
    stem = Path(source).stem if source else "doc"
    allowed = meta.get("allowed_roles") or ["employee"]
    return Document(
        doc_id=str(meta.get("doc_id", stem)),
        title=str(meta.get("title", stem.replace("-", " ").title())),
        body=body.strip(),
        allowed_roles=tuple(allowed),
        classification=str(meta.get("classification", "internal")),
        sensitive=bool(meta.get("sensitive", False)),
        source=source,
        content_hash=hashlib.sha256(raw.encode()).hexdigest(),
        sensitive_sections=tuple(meta.get("sensitive_sections") or ()),
    )


def load_documents(directory: Path) -> list[Document]:
    docs = [parse_document(p.read_text(encoding="utf-8"), str(p)) for p in sorted(Path(directory).glob("*.md"))]
    ids = [d.doc_id for d in docs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate doc_id in corpus")
    return docs


def _sections(body: str) -> list[tuple[str, str]]:
    section, buf, out = "Overview", [], []
    for line in body.splitlines():
        m = _HEADING.match(line)
        if m:
            if "".join(buf).strip():
                out.append((section, "\n".join(buf)))
            section, buf = m.group(2), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        out.append((section, "\n".join(buf)))
    return out


def pack_sentences(sentences: list[str], size: int, overlap: int) -> list[list[str]]:
    """Greedy sentence packing into ~``size``-word windows, carrying ~``overlap`` words forward."""
    groups: list[list[str]] = []
    cur: list[str] = []
    cur_w = 0
    for s in sentences:
        w = len(s.split())
        if cur and cur_w + w > size:
            groups.append(cur)
            carry: list[str] = []
            cw = 0
            for prev in reversed(cur):
                pw = len(prev.split())
                if cw + pw > overlap:
                    break
                carry.insert(0, prev)
                cw += pw
            cur, cur_w = carry, cw
        cur.append(s)
        cur_w += w
    if cur:
        groups.append(cur)
    return groups


def chunk_document(doc: Document, size: int = 80, overlap: int = 20) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section, text in _sections(doc.body):
        flat = re.sub(r"\s*\n\s*", " ", clean_markdown(text)).strip()
        for group in pack_sentences(split_sentences(flat), size, overlap):
            body = " ".join(group)
            pos = len(chunks)
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}#{pos:03d}",
                doc_id=doc.doc_id,
                title=doc.title,
                section=section,
                text=body,
                index_text=f"{doc.title}. {section}. {body}",
                position=pos,
                allowed_roles=doc.allowed_roles,
                classification=doc.classification,
                sensitive=doc.sensitive or section in doc.sensitive_sections,
                source=doc.source,
                content_hash=doc.content_hash,
            ))
    return chunks


class Ingestor:
    def __init__(self, index: Index, embedder: Embedder, chunk_size: int = 80, chunk_overlap: int = 20):
        self.index = index
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest_documents(self, docs: list[Document], prune: bool = True) -> IngestReport:
        t0 = time.perf_counter()
        report = IngestReport()
        seen = set()
        for doc in docs:
            seen.add(doc.doc_id)
            entry = self.index.manifest.get(doc.doc_id)
            if entry and entry["hash"] == doc.content_hash and entry["vectors"]:
                report.unchanged.append(doc.doc_id)
                continue
            chunks = chunk_document(doc, self.chunk_size, self.chunk_overlap)
            try:
                vectors = self.embedder.embed([c.index_text for c in chunks]) if chunks else None
            except EmbedderUnavailable:
                vectors = None  # degrade: index lexically now, embed on the next ingest
                report.lexical_only.append(doc.doc_id)
            self.index.upsert_document(doc.doc_id, chunks, vectors, doc.content_hash)
            (report.updated if entry else report.added).append(doc.doc_id)
        if prune:
            for gone in [d for d in self.index.manifest if d not in seen]:
                self.index.remove_document(gone)
                report.removed.append(gone)
        report.chunks = len(self.index)
        report.seconds = round(time.perf_counter() - t0, 4)
        return report

    def ingest_directory(self, directory: Path, prune: bool = True) -> IngestReport:
        return self.ingest_documents(load_documents(directory), prune=prune)
