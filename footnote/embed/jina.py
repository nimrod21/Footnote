"""Jina v3 embeddings over HTTP. No local models, no PyTorch.

Uses asymmetric task prefixes (retrieval.passage vs retrieval.query) — worth
real recall and free. Every call is cached; cumulative token usage is tracked
so quota consumption is measurable, not guessed.
"""

from __future__ import annotations

import time

import httpx

from footnote.config import Secrets
from footnote.embed.base import EmbeddingCache

API_URL = "https://api.jina.ai/v1/embeddings"
BATCH = 64
RETRIES = 5


class JinaEmbeddings:
    name = "jina"
    model_id = "jina-embeddings-v3"
    dimensions = 1024

    def __init__(self, secrets: Secrets | None = None, cache: EmbeddingCache | None = None):
        self.secrets = secrets or Secrets()
        self.cache = cache or EmbeddingCache()
        self.tokens_used = 0  # this process, cache misses only

    # -- public --------------------------------------------------------------

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, task="retrieval.passage")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], task="retrieval.query")[0]

    # -- internals -----------------------------------------------------------

    def _embed(self, texts: list[str], task: str) -> list[list[float]]:
        keys = [self.cache.key(t, self.model_id, task) for t in texts]
        out: list[list[float] | None] = [self.cache.get(k) for k in keys]
        missing = [i for i, v in enumerate(out) if v is None]

        for start in range(0, len(missing), BATCH):
            idx = missing[start : start + BATCH]
            vectors = self._call_api([texts[i] for i in idx], task)
            self.cache.put_many([(keys[i], v) for i, v in zip(idx, vectors)])
            for i, v in zip(idx, vectors):
                out[i] = v
        return out  # type: ignore[return-value]

    def _call_api(self, texts: list[str], task: str) -> list[list[float]]:
        payload = {"model": self.model_id, "task": task, "input": texts}
        headers = {"Authorization": f"Bearer {self.secrets.jina_api_key}"}
        delay = 2.0
        for attempt in range(RETRIES):
            resp = httpx.post(API_URL, json=payload, headers=headers, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                self.tokens_used += data.get("usage", {}).get("total_tokens", 0)
                return [d["embedding"] for d in data["data"]]
            if resp.status_code in (429, 500, 502, 503) and attempt < RETRIES - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise RuntimeError(f"Jina API {resp.status_code}: {resp.text[:300]}")
        raise RuntimeError("unreachable")  # pragma: no cover


def get_provider(name: str) -> JinaEmbeddings:
    """Provider factory. 'local' lands with the [local] extra if ever needed."""
    if name == "jina":
        return JinaEmbeddings()
    raise ValueError(f"unknown embedding provider: {name}")
