"""Embedding provider protocol + disk cache.

The cache is what keeps ablation runs affordable on a free quota: re-indexing
after a chunking change never re-embeds text that hasn't changed.
"""

from __future__ import annotations

import hashlib
import sqlite3
from array import array
from pathlib import Path
from typing import Protocol

CACHE_PATH = Path("data/cache/embeddings.db")


class EmbeddingProvider(Protocol):
    name: str
    model_id: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class EmbeddingCache:
    """sha256(text + model + task) -> vector, in a single SQLite file."""

    def __init__(self, path: Path = CACHE_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, dim INTEGER, vec BLOB)"
        )

    @staticmethod
    def key(text: str, model_id: str, task: str) -> str:
        return hashlib.sha256(f"{model_id}\x00{task}\x00{text}".encode()).hexdigest()

    def get(self, key: str) -> list[float] | None:
        row = self.db.execute("SELECT dim, vec FROM emb WHERE key=?", (key,)).fetchone()
        if not row:
            return None
        vec = array("f")
        vec.frombytes(row[1])
        return list(vec)

    def put(self, key: str, vector: list[float]) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO emb VALUES (?,?,?)",
            (key, len(vector), array("f", vector).tobytes()),
        )
        self.db.commit()

    def put_many(self, items: list[tuple[str, list[float]]]) -> None:
        self.db.executemany(
            "INSERT OR REPLACE INTO emb VALUES (?,?,?)",
            [(k, len(v), array("f", v).tobytes()) for k, v in items],
        )
        self.db.commit()
