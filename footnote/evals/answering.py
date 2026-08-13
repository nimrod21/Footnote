"""Answer-quality and refusal suites — the LLM-dependent half of M6.

Scores, all mechanical (no judge here):
- answerable set: citation precision/recall against gold articles, verbatim
  integrity (must be 1.0 by construction), wrongly-refused rate
- negatives: the refusal confusion matrix — false-answer rate is the headline
- traps: fabrication check — did it invent a provision that doesn't exist?
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from footnote.answer.generate import Answerer
from footnote.evals.golden import GoldenItem
from footnote.evals.retrieval import matches


@dataclass
class AnswerRow:
    qid: str
    source: str
    expect: str
    verdict: str
    gate_fired: str | None
    n_citations: int
    cited_gold: int  # citations matching a gold article
    cited_labels: list[str]
    latency_ms: int
    error: str | None = None


def run_item(answerer: Answerer, item: GoldenItem, skip_scope_gate: bool) -> AnswerRow:
    try:
        r = answerer.ask(item.question, skip_scope_gate=skip_scope_gate)
    except RuntimeError as e:
        return AnswerRow(item.qid, item.source, item.expect, "error", None, 0, 0, [],
                         0, error=str(e)[:200])
    a = r.answer
    cited_gold = sum(
        1 for c in a.citations if any(matches(c.provision_id, g) for g in item.gold_articles)
    )
    return AnswerRow(
        qid=item.qid, source=item.source, expect=item.expect,
        verdict=a.verdict, gate_fired=r.gate_fired,
        n_citations=len(a.citations), cited_gold=cited_gold,
        cited_labels=[c.citation_label for c in a.citations],
        latency_ms=r.llm.latency_ms if r.llm else 0,
    )


def aggregate_answerable(rows: list[AnswerRow]) -> dict:
    rows = [r for r in rows if not r.error]
    n = len(rows)
    answered = [r for r in rows if r.verdict == "answered"]
    with_cites = [r for r in answered if r.n_citations]
    return {
        "n": n,
        "answered_rate": round(len(answered) / n, 4) if n else 0,
        "over_refusal_rate": round(sum(r.verdict != "answered" for r in rows) / n, 4) if n else 0,
        "citation_precision": round(
            sum(r.cited_gold for r in with_cites) / max(1, sum(r.n_citations for r in with_cites)), 4),
        "any_gold_cited_rate": round(
            sum(r.cited_gold > 0 for r in answered) / max(1, len(answered)), 4),
    }


def aggregate_negatives(rows: list[AnswerRow]) -> dict:
    rows = [r for r in rows if not r.error]
    out = {}
    for kind in ("negative:out_of_scope", "negative:unanswerable", "negative:trap"):
        sub = [r for r in rows if r.source == kind]
        if not sub:
            continue
        correct = sum(
            1 for r in sub
            if (r.expect == "out_of_scope" and r.verdict in ("out_of_scope", "refused"))
            or (r.expect == "refused" and r.verdict in ("refused", "out_of_scope"))
        )
        out[kind.split(":")[1]] = {
            "n": len(sub),
            "correct_refusal_rate": round(correct / len(sub), 4),
            "false_answer_rate": round(sum(r.verdict == "answered" for r in sub) / len(sub), 4),
        }
    # fabrication: an "answered" trap citing anything is an invented-provision answer
    traps = [r for r in rows if r.source == "negative:trap"]
    out["fabrication_rate"] = round(
        sum(r.verdict == "answered" and r.n_citations > 0 for r in traps) / max(1, len(traps)), 4)
    return out


def run_suites(answerer: Answerer, items: list[GoldenItem], sleep_s: float = 0.5):
    rows = []
    for item in items:
        # negatives keep the scope gate on (it IS the thing under test for out_of_scope);
        # answerable items skip it to isolate answer quality from scope classification.
        skip = not item.source.startswith("negative")
        rows.append(run_item(answerer, item, skip_scope_gate=skip))
        time.sleep(sleep_s)
    return rows


def to_dicts(rows: list[AnswerRow]) -> list[dict]:
    return [asdict(r) for r in rows]
