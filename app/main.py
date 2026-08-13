"""Footnote REST API. Typed models -> OpenAPI at /docs.

Error contract: a refusal is a 200 with verdict='refused' — never an error.
400 bad input, 404 unknown provision/trace, 429 rate-limited, 503 upstream
provider failure. Every /query writes the same trace row as CLI and MCP.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from footnote.config import RunConfig
from footnote.models import DISCLAIMER

app = FastAPI(
    title="Footnote",
    description="Grounded answers over the GDPR and the EU AI Act. Every claim "
    "cites the exact provision; unanswerable questions are refused. "
    "**Not legal advice.**",
    version="0.1.0",
)

_state: dict[str, Any] = {}


def _answerer():
    if "answerer" not in _state:
        from footnote.answer.generate import Answerer

        _state["config"] = RunConfig()
        _state["answerer"] = Answerer(_state["config"])
    return _state["answerer"]


# -- rate limiting: per-IP sliding window + global daily cap ------------------

RATE_WINDOW_S, RATE_MAX = 60, 10  # 10 requests/min/IP
DAILY_LLM_CAP = 40  # global: the OpenRouter free tier is 50/day, leave headroom
_hits: dict[str, deque] = {}
_llm_calls_today = {"day": "", "n": 0}


def _rate_check(ip: str, llm: bool) -> None:
    now = time.time()
    dq = _hits.setdefault(ip, deque())
    while dq and dq[0] < now - RATE_WINDOW_S:
        dq.popleft()
    if len(dq) >= RATE_MAX:
        raise HTTPException(429, "Rate limit: 10 requests per minute per IP.")
    dq.append(now)
    if llm:
        day = time.strftime("%Y-%m-%d")
        if _llm_calls_today["day"] != day:
            _llm_calls_today.update(day=day, n=0)
        if _llm_calls_today["n"] >= DAILY_LLM_CAP:
            raise HTTPException(
                429, "Daily free-tier answer budget exhausted; try /search, or come back tomorrow.")
        _llm_calls_today["n"] += 1


# -- models -------------------------------------------------------------------


class QueryIn(BaseModel):
    question: str = Field(min_length=5, max_length=500)
    instrument: Literal["gdpr", "ai_act"] | None = None
    agent: bool = False
    max_hops: int = Field(4, ge=1, le=6)


class SearchIn(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    instrument: Literal["gdpr", "ai_act"] | None = None
    type: Literal["article", "recital", "annex", "definition"] | None = None
    top_k: int = Field(10, ge=1, le=25)


class CitationOut(BaseModel):
    provision_id: str
    citation_label: str
    quote: str
    eurlex_url: str
    instrument: str


class AnswerOut(BaseModel):
    verdict: Literal["answered", "refused", "out_of_scope"]
    text: str | None
    citations: list[CitationOut]
    confidence: float
    refusal_reason: str | None
    disclaimer: str
    cost_usd: float
    latency_ms: int
    trace_id: str
    agent_hops: int | None = None


class ProvisionOut(BaseModel):
    provision_id: str
    citation_label: str
    instrument: str
    type: str
    heading: str | None
    text: str
    eurlex_url: str
    cross_refs: list[str]


class SearchHit(ProvisionOut):
    score: float


# -- endpoints ----------------------------------------------------------------


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.post("/query", response_model=AnswerOut)
def query(q: QueryIn, request: Request) -> AnswerOut:
    _rate_check(request.client.host if request.client else "?", llm=True)
    answerer = _answerer()
    cfg = _state["config"]
    cfg.max_hops = q.max_hops
    from footnote.trace.db import record

    try:
        if q.agent:
            from footnote.agent.run import ask_with_agent

            ar = ask_with_agent(q.question, cfg, answerer=answerer)
            r, trace = ar.ask, ar.trace
        else:
            r, trace = answerer.ask(q.question, instrument=q.instrument), None
    except RuntimeError as e:
        raise HTTPException(503, f"Upstream model unavailable: {e}") from e

    qid = record("api", q.question, r, cfg, agent_trace=trace)
    a = r.answer
    return AnswerOut(
        verdict=a.verdict, text=a.text,
        citations=[CitationOut(**c.model_dump()) for c in a.citations],
        confidence=a.confidence, refusal_reason=a.refusal_reason,
        disclaimer=a.disclaimer,
        cost_usd=r.llm.cost_usd if r.llm else 0.0,
        latency_ms=r.llm.latency_ms if r.llm else 0,
        trace_id=qid, agent_hops=len(trace.hops) if trace else None,
    )


@app.post("/search", response_model=list[SearchHit])
def search(s: SearchIn, request: Request) -> list[SearchHit]:
    _rate_check(request.client.host if request.client else "?", llm=False)
    ret = _answerer().retriever.search(
        s.query, instrument=s.instrument, type=s.type, top_n=s.top_k
    )
    return [
        SearchHit(score=round(r.score, 4), **{
            k: getattr(r.provision, k) for k in ProvisionOut.model_fields
        })
        for r in ret.results
    ]


@app.get("/provisions/{provision_id}", response_model=ProvisionOut)
def provision(provision_id: str) -> ProvisionOut:
    p = _answerer().retriever.lookup(provision_id)
    if p is None:
        raise HTTPException(404, f"No provision {provision_id!r}.")
    return ProvisionOut(**{k: getattr(p, k) for k in ProvisionOut.model_fields})


@app.get("/instruments")
def instruments() -> list[dict]:
    from footnote.corpus.registry import INSTRUMENTS

    retriever = _answerer().retriever
    counts: dict[str, int] = {}
    for p in retriever.provisions.values():
        counts[p.instrument] = counts.get(p.instrument, 0) + 1
    return [
        {**inst.model_dump(), "provision_count": counts.get(iid, 0), "disclaimer": DISCLAIMER}
        for iid, inst in INSTRUMENTS.items()
    ]


@app.get("/traces/{trace_id}")
def trace(trace_id: str) -> JSONResponse:
    from footnote.trace import db

    t = db.get(trace_id)
    if t is None:
        raise HTTPException(404, "Unknown trace id.")
    return JSONResponse(t)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")
