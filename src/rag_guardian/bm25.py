"""Incremental Okapi BM25 over an inverted index, with ACL pre-filtering."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Collection


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._postings: dict[str, dict[str, int]] = {}
        self._doc_len: dict[str, int] = {}
        self._doc_terms: dict[str, list[str]] = {}
        self._total_len = 0

    def __len__(self) -> int:
        return len(self._doc_len)

    @property
    def avgdl(self) -> float:
        return self._total_len / len(self._doc_len) if self._doc_len else 0.0

    def add(self, doc_id: str, terms: list[str]) -> None:
        if doc_id in self._doc_len:
            self.remove(doc_id)
        counts = Counter(terms)
        for term, tf in counts.items():
            self._postings.setdefault(term, {})[doc_id] = tf
        self._doc_len[doc_id] = len(terms)
        self._doc_terms[doc_id] = list(counts)
        self._total_len += len(terms)

    def remove(self, doc_id: str) -> None:
        if doc_id not in self._doc_len:
            return
        for term in self._doc_terms.pop(doc_id):
            posting = self._postings.get(term)
            if posting is not None:
                posting.pop(doc_id, None)
                if not posting:
                    del self._postings[term]
        self._total_len -= self._doc_len.pop(doc_id)

    def df(self, term: str) -> int:
        return len(self._postings.get(term, ()))

    def idf(self, term: str) -> float:
        n = len(self._doc_len)
        df = self.df(term)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def vocabulary(self):
        """Live keys view: O(1) membership tests, no copy."""
        return self._postings.keys()

    def search(self, terms: list[str], k: int, allowed: Collection[str] | None = None) -> list[tuple[str, float]]:
        if not self._doc_len:
            return []
        scores: dict[str, float] = {}
        avgdl = self.avgdl or 1.0
        for term in set(terms):
            posting = self._postings.get(term)
            if not posting:
                continue
            idf = self.idf(term)
            for doc_id, tf in posting.items():
                if allowed is not None and doc_id not in allowed:
                    continue
                dl = self._doc_len[doc_id]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / avgdl)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]
