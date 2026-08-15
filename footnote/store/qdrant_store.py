"""Qdrant wrapper: cloud when QDRANT_URL is set, embedded local file otherwise.

One collection holds both regulations; `instrument` and `type` are payload-indexed
so filtered search (one regulation only, binding text only, a single provision)
is a query parameter, not an afterthought.
"""

from __future__ import annotations

import uuid

from qdrant_client import QdrantClient, models as qm

from footnote.config import Secrets
from footnote.models import Provision

_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # uuid5 namespace


def point_id(provision_id: str) -> str:
    """Deterministic UUID per provision — upserts stay idempotent."""
    return str(uuid.uuid5(_NS, provision_id))


def collection_name(model_id: str, strategy: str) -> str:
    return f"footnote__{model_id.replace('-', '_')}__{strategy}"


_embedded_client: QdrantClient | None = None


class QdrantStore:
    def __init__(self, secrets: Secrets | None = None):
        global _embedded_client
        s = secrets or Secrets()
        if s.qdrant_url:
            self.client = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key, timeout=60)
            self.mode = "cloud"
        else:
            # embedded mode holds a file lock — one client per process, shared
            if _embedded_client is None:
                _embedded_client = QdrantClient(path="data/qdrant")
            self.client = _embedded_client
            self.mode = "embedded"

    def ensure_collection(self, name: str, dimensions: int) -> None:
        if not self.client.collection_exists(name):
            self.client.create_collection(
                name, vectors_config=qm.VectorParams(size=dimensions, distance=qm.Distance.COSINE)
            )
            for field in ("instrument", "type"):
                self.client.create_payload_index(
                    name, field_name=field, field_schema=qm.PayloadSchemaType.KEYWORD
                )

    def upsert(self, name: str, provisions: list[Provision], vectors: list[list[float]]) -> None:
        points = [
            qm.PointStruct(
                id=point_id(p.provision_id),
                vector=v,
                payload=p.model_dump(),
            )
            for p, v in zip(provisions, vectors)
        ]
        for start in range(0, len(points), 256):
            self.client.upsert(name, points[start : start + 256])

    def search(
        self,
        name: str,
        vector: list[float],
        top_k: int = 50,
        instrument: str | None = None,
        type: str | None = None,
    ) -> list[tuple[float, dict]]:
        must: list[qm.FieldCondition] = []
        if instrument:
            must.append(qm.FieldCondition(key="instrument", match=qm.MatchValue(value=instrument)))
        if type:
            must.append(qm.FieldCondition(key="type", match=qm.MatchValue(value=type)))
        hits = self.client.query_points(
            name,
            query=vector,
            limit=top_k,
            query_filter=qm.Filter(must=must) if must else None,
            with_payload=True,
        ).points
        return [(h.score, h.payload) for h in hits]

    def count(self, name: str) -> int:
        return self.client.count(name).count
