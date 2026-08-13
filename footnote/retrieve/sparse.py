"""BM25 over provision chunks.

The sparse half matters more here than in typical RAG: statutory text is full
of exact terms of art ("legitimate interest", "high-risk AI system") where
embeddings blur distinctions the statute treats as decisive.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from footnote.models import Provision

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class SparseIndex:
    def __init__(self, chunks: list[Provision]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def search(
        self,
        query: str,
        top_k: int = 50,
        instrument: str | None = None,
        type: str | None = None,
    ) -> list[tuple[float, Provision]]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(scores, self.chunks), key=lambda x: -x[0])
        out = []
        for score, chunk in ranked:
            if score <= 0:
                break
            if instrument and chunk.instrument != instrument:
                continue
            if type and chunk.type != type:
                continue
            out.append((float(score), chunk))
            if len(out) >= top_k:
                break
        return out
