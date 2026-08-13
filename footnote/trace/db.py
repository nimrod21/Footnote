"""Query traces: one SQLite row per question, whatever the entry point.

CLI, REST, MCP, and agent all funnel through record() — one code path.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

TRACE_PATH = Path("data/traces.db")

_SCHEMA = """CREATE TABLE IF NOT EXISTS traces (
    query_id TEXT PRIMARY KEY,
    ts TEXT, source TEXT, question TEXT,
    verdict TEXT, gate_fired TEXT, confidence REAL,
    config_hash TEXT, generation_model TEXT, prompt_version TEXT,
    retrieved_ids TEXT, citation_labels TEXT,
    prompt_tokens INTEGER, completion_tokens INTEGER,
    cost_usd REAL, latency_ms INTEGER,
    agent_hops INTEGER, agent_terminated_by TEXT, agent_trace TEXT
)"""


def _db() -> sqlite3.Connection:
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(TRACE_PATH)
    db.execute(_SCHEMA)
    return db


def record(source: str, question: str, ask_result, config, agent_trace=None) -> str:
    """Persist one query. Returns the query_id."""
    a = ask_result.answer
    llm = ask_result.llm
    ret = ask_result.retrieval
    qid = str(uuid.uuid4())
    row = (
        qid, time.strftime("%Y-%m-%dT%H:%M:%S"), source, question,
        a.verdict, ask_result.gate_fired, a.confidence,
        config.config_hash(),
        llm.model if llm else None,
        ask_result.prompt_version,
        json.dumps([r.provision.provision_id for r in ret.results]) if ret else "[]",
        json.dumps([c.citation_label for c in a.citations]),
        llm.prompt_tokens if llm else 0,
        llm.completion_tokens if llm else 0,
        llm.cost_usd if llm else 0.0,
        llm.latency_ms if llm else 0,
        len(agent_trace.hops) if agent_trace else None,
        agent_trace.terminated_by if agent_trace else None,
        json.dumps([{
            "n": h.n, "tool": h.tool, "args": h.args, "reasoning": h.reasoning,
            "observation": h.observation, "provision_ids": h.provision_ids,
        } for h in agent_trace.hops]) if agent_trace else None,
    )
    db = _db()
    db.execute(f"INSERT INTO traces VALUES ({','.join('?' * len(row))})", row)
    db.commit()
    return qid


def recent(limit: int = 20) -> list[dict]:
    db = _db()
    db.row_factory = sqlite3.Row
    rows = db.execute("SELECT * FROM traces ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def get(query_id: str) -> dict | None:
    db = _db()
    db.row_factory = sqlite3.Row
    row = db.execute("SELECT * FROM traces WHERE query_id=?", (query_id,)).fetchone()
    return dict(row) if row else None


def summary() -> dict:
    db = _db()
    row = db.execute(
        "SELECT COUNT(*), SUM(cost_usd), AVG(latency_ms), "
        "SUM(CASE WHEN verdict='refused' THEN 1 ELSE 0 END), "
        "SUM(prompt_tokens + completion_tokens) FROM traces"
    ).fetchone()
    n = row[0] or 0
    return {
        "queries": n,
        "total_cost_usd": round(row[1] or 0.0, 6),
        "avg_latency_ms": int(row[2] or 0),
        "refusal_rate": round((row[3] or 0) / n, 3) if n else 0.0,
        "total_tokens": row[4] or 0,
    }
