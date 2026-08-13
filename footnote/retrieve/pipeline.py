"""Retrieval orchestrator.

    query -> [citation shortcut] -> dense + sparse -> RRF -> rerank -> results

Every stage toggles via RunConfig — "reranker off" is a config flag, not a code
edit; that's what makes the ablation table possible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from footnote.config import RunConfig
from footnote.corpus.load import load_chunks, load_provisions
from footnote.corpus.registry import SHORT_NAME
from footnote.embed.jina import get_provider
from footnote.models import Provision
from footnote.retrieve.fusion import rrf
from footnote.retrieve.rerank import JinaReranker
from footnote.retrieve.sparse import SparseIndex
from footnote.store.qdrant_store import QdrantStore, collection_name


@dataclass
class RetrievalResult:
    provision: Provision
    score: float  # final stage score (rerank if enabled, else fused)
    dense_score: float | None = None
    sparse_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    via: str = "hybrid"  # "hybrid" | "citation"


@dataclass
class Retrieval:
    results: list[RetrievalResult]
    confidence: float  # top-1 final score; the refusal signal
    stages: dict = field(default_factory=dict)  # latency per stage, filled by caller

    def top_gap(self) -> float:
        if len(self.results) < 2:
            return 0.0
        return self.results[0].score - self.results[1].score


# "Article 6(1)(f) GDPR", "Art. 22", "Recital 47 GDPR", "Annex III AI Act", "Annex III(4)"
_CITE = re.compile(
    r"\b(?:(Art(?:icle)?s?\.?)\s+(\d{1,3})\s*(?:\((\d{1,2})\))?\s*(?:\(([a-z]{1,3})\))?"
    r"|(Recitals?)\s+(\d{1,3})"
    r"|(Annex(?:es)?)\s+([IVXLC]+)\s*(?:\((\d{1,2})\))?\s*(?:\(([a-z]{1,3})\))?)"
    r"(?:\s+(?:of\s+the\s+)?(GDPR|AI\s*Act))?",
    re.IGNORECASE,
)


def parse_citation(query: str) -> list[str]:
    """Direct citation references in a query -> candidate provision ids."""
    ids: list[str] = []
    for m in _CITE.finditer(query):
        inst_raw = (m.group(11) or "").lower().replace(" ", "")
        instruments = ["gdpr"] if inst_raw == "gdpr" else ["ai_act"] if inst_raw == "aiact" else ["gdpr", "ai_act"]
        for iid in instruments:
            if m.group(1):  # article
                pid = f"{iid}:art:{m.group(2)}"
                if m.group(3):
                    pid += f":{m.group(3)}"
                if m.group(4):
                    pid += f":{m.group(4)}"
                ids.append(pid)
            elif m.group(5):  # recital
                ids.append(f"{iid}:rec:{m.group(6)}")
            elif m.group(7):  # annex
                pid = f"{iid}:anx:{m.group(8).upper()}"
                if m.group(9):
                    pid += f":{m.group(9)}"
                if m.group(10):
                    pid += f":{m.group(10)}"
                ids.append(pid)
    return ids


class Retriever:
    def __init__(self, config: RunConfig | None = None):
        self.config = config or RunConfig()
        self.provisions = load_provisions(tuple(self.config.corpus))
        self.chunks = load_chunks(self.config.chunk_strategy, tuple(self.config.corpus))
        self.sparse = SparseIndex(self.chunks) if self.config.sparse_enabled else None
        self.embedder = get_provider(self.config.embedding_provider)
        self.store = QdrantStore()
        self.collection = collection_name(self.embedder.model_id, self.config.chunk_strategy)
        self.reranker = JinaReranker() if self.config.rerank_enabled else None
        self._by_id = {c.provision_id: c for c in self.chunks}

    def lookup(self, provision_id: str) -> Provision | None:
        return self.provisions.get(provision_id)

    def search(
        self,
        query: str,
        instrument: str | None = None,
        type: str | None = None,
        top_n: int | None = None,
    ) -> Retrieval:
        cfg = self.config
        top_n = top_n or cfg.rerank_top_n

        # Citation shortcut: an explicit reference beats any similarity search.
        cited = [
            self.provisions[pid]
            for pid in parse_citation(query)
            if pid in self.provisions and (not instrument or pid.startswith(instrument))
        ]
        if cited:
            results = [
                RetrievalResult(provision=p, score=1.0, via="citation") for p in cited[:top_n]
            ]
            return Retrieval(results=results, confidence=1.0)

        dense_scores: dict[str, float] = {}
        sparse_scores: dict[str, float] = {}
        if cfg.dense_enabled:
            vec = self.embedder.embed_query(query)
            for score, payload in self.store.search(
                self.collection, vec, cfg.top_k_dense, instrument=instrument, type=type
            ):
                dense_scores[payload["provision_id"]] = score
        if self.sparse is not None:
            for score, chunk in self.sparse.search(
                query, cfg.top_k_sparse, instrument=instrument, type=type
            ):
                sparse_scores[chunk.provision_id] = score

        fused = rrf(
            [r for r in (list(dense_scores), list(sparse_scores)) if r], k=cfg.fusion_k
        )
        candidates = [pid for pid, _ in fused[: max(cfg.top_k_dense, cfg.top_k_sparse)]]
        fused_map = dict(fused)

        if self.reranker is not None and candidates:
            docs = [self._doc_text(pid) for pid in candidates]
            order = self.reranker.rerank(query, docs, top_n)
            picked = [(candidates[i], score) for i, score in order]
        else:
            picked = [(pid, fused_map[pid]) for pid in candidates[:top_n]]

        results = [
            RetrievalResult(
                provision=self._by_id.get(pid) or self.provisions[pid],
                score=score,
                dense_score=dense_scores.get(pid),
                sparse_score=sparse_scores.get(pid),
                fused_score=fused_map.get(pid),
                rerank_score=score if self.reranker is not None else None,
            )
            for pid, score in picked
        ]
        return Retrieval(results=results, confidence=results[0].score if results else 0.0)

    def _doc_text(self, pid: str) -> str:
        p = self._by_id.get(pid) or self.provisions[pid]
        head = f"{p.citation_label}" + (f" — {p.heading}" if p.heading else "")
        return f"{head}: {p.text}"
