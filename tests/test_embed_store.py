"""Cache and store logic — no network required."""
from pathlib import Path

from footnote.embed.base import EmbeddingCache
from footnote.store.qdrant_store import collection_name, point_id


def test_cache_roundtrip(tmp_path: Path):
    c = EmbeddingCache(tmp_path / "e.db")
    k = c.key("some text", "model-x", "retrieval.passage")
    assert c.get(k) is None
    c.put(k, [0.1, 0.2, 0.3])
    got = c.get(k)
    assert got is not None and len(got) == 3 and abs(got[1] - 0.2) < 1e-6


def test_cache_key_distinguishes_task_and_model():
    k = EmbeddingCache.key
    t = "same text"
    assert len({k(t, "m1", "retrieval.passage"), k(t, "m2", "retrieval.passage"),
                k(t, "m1", "retrieval.query")}) == 3


def test_point_id_deterministic():
    assert point_id("gdpr:art:6:1:f") == point_id("gdpr:art:6:1:f")
    assert point_id("gdpr:art:6:1:f") != point_id("gdpr:art:6:1:e")


def test_collection_name():
    assert collection_name("jina-embeddings-v3", "provision") == \
        "footnote__jina_embeddings_v3__provision"
