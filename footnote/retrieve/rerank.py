"""Jina reranker over the fused candidates. Highest-value single stage in most
RAG pipelines; the ablation proving (or disproving) that here is a README row."""

from __future__ import annotations

import time

import httpx

from footnote.config import Secrets

API_URL = "https://api.jina.ai/v1/rerank"
RETRIES = 4


class JinaReranker:
    model_id = "jina-reranker-v2-base-multilingual"

    def __init__(self, secrets: Secrets | None = None):
        self.secrets = secrets or Secrets()
        self.units_used = 0

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        """Returns (original_index, relevance_score), best first."""
        payload = {
            "model": self.model_id,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
        }
        headers = {"Authorization": f"Bearer {self.secrets.jina_api_key}"}
        delay = 2.0
        for attempt in range(RETRIES):
            resp = httpx.post(API_URL, json=payload, headers=headers, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                self.units_used += data.get("usage", {}).get("total_tokens", 0)
                return [(r["index"], r["relevance_score"]) for r in data["results"]]
            if resp.status_code in (429, 500, 502, 503) and attempt < RETRIES - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise RuntimeError(f"Jina rerank {resp.status_code}: {resp.text[:300]}")
        raise RuntimeError("unreachable")  # pragma: no cover
