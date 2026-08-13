"""Footnote MCP server — GDPR + EU AI Act as tools for any agent.

Refusals are normal results, never errors: a calling agent must distinguish
"the regulations don't answer this" from "the tool broke". The disclaimer
travels in every answer payload. Every call writes the same trace row a CLI
query does.
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

from footnote.config import RunConfig

server = MCPServer(
    name="footnote",
    instructions=(
        "Grounded answers over the GDPR and the EU AI Act. Use search for "
        "open questions, lookup for exact provisions, answer for a cited "
        "answer (verdict may be 'refused' or 'out_of_scope' - that is a "
        "correct result, not an error). Not legal advice."
    ),
)
_state: dict[str, Any] = {}  # lazy singletons: retriever/answerer build on first call


def _answerer():
    if "answerer" not in _state:
        from footnote.answer.generate import Answerer

        _state["config"] = RunConfig()
        _state["answerer"] = Answerer(_state["config"])
    return _state["answerer"]


@server.tool()
def search(query: str, instrument: str | None = None, type: str | None = None,
           top_k: int = 10) -> list:
    """Hybrid search (dense + BM25 + rerank) over the GDPR and the EU AI Act.
    Returns provisions with legal citation labels and EUR-Lex links.
    instrument: gdpr|ai_act; type: article|recital|annex|definition."""
    return _dispatch("search", {"query": query, "instrument": instrument,
                                "type": type, "top_k": top_k})


@server.tool()
def lookup(reference: str) -> dict:
    """Fetch one provision by internal id ('gdpr:art:6:1:f', 'ai_act:anx:III:4')
    or citation ('Article 6(1)(f) GDPR')."""
    return _dispatch("lookup", {"reference": reference})


@server.tool()
def answer(question: str, instrument: str | None = None, agent: bool = False) -> dict:
    """Answer a GDPR / AI Act question, grounded in retrieved provisions with
    verified verbatim quotes. May refuse - that is a correct result, not an
    error. agent=true enables multi-hop research across cross-references."""
    return _dispatch("answer", {"question": question, "instrument": instrument,
                                "agent": agent})


@server.tool()
def list_instruments() -> list:
    """List the indexed regulations with version dates and provision counts."""
    return _dispatch("list_instruments", {})

def _dispatch(name: str, args: dict) -> dict | list:
    answerer = _answerer()
    retriever = answerer.retriever
    cfg = _state["config"]

    if name == "search":
        ret = retriever.search(
            args["query"], instrument=args.get("instrument"),
            type=args.get("type"), top_n=int(args.get("top_k", 10)),
        )
        return [
            {
                "provision_id": r.provision.provision_id,
                "citation_label": r.provision.citation_label,
                "heading": r.provision.heading,
                "text": r.provision.text,
                "score": round(r.score, 4),
                "eurlex_url": r.provision.eurlex_url,
            }
            for r in ret.results
        ]

    if name == "lookup":
        from footnote.retrieve.pipeline import parse_citation

        ref = str(args["reference"]).strip()
        p = retriever.lookup(ref)
        if p is None:
            for pid in parse_citation(ref):
                if p := retriever.lookup(pid):
                    break
        if p is None:
            return {"found": False, "reference": ref,
                    "note": "No such provision. GDPR has 99 articles; the AI Act has 113."}
        return {"found": True, **p.model_dump()}

    if name == "answer":
        from footnote.trace.db import record

        if args.get("agent"):
            from footnote.agent.run import ask_with_agent

            ar = ask_with_agent(str(args["question"]), cfg, answerer=answerer)
            r, trace = ar.ask, ar.trace
        else:
            r, trace = answerer.ask(
                str(args["question"]), instrument=args.get("instrument")
            ), None
        qid = record("mcp", str(args["question"]), r, cfg, agent_trace=trace)
        a = r.answer
        return {
            "verdict": a.verdict,
            "text": a.text,
            "citations": [c.model_dump() for c in a.citations],
            "confidence": a.confidence,
            "refusal_reason": a.refusal_reason,
            "disclaimer": a.disclaimer,
            "cost_usd": r.llm.cost_usd if r.llm else 0.0,
            "latency_ms": r.llm.latency_ms if r.llm else 0,
            "trace_id": qid,
            "agent_hops": len(trace.hops) if trace else None,
        }

    if name == "list_instruments":
        from footnote.corpus.registry import INSTRUMENTS

        counts: dict[str, int] = {}
        for p in retriever.provisions.values():
            counts[p.instrument] = counts.get(p.instrument, 0) + 1
        return [
            {**inst.model_dump(), "provision_count": counts.get(iid, 0)}
            for iid, inst in INSTRUMENTS.items()
        ]

    return {"error": f"unknown tool {name}"}


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
