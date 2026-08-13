"""Retrieval metrics — mechanical, no LLM involved.

A retrieved provision matches a gold article if it IS the gold id or sits
underneath it (gold "gdpr:art:33" matches retrieved "gdpr:art:33:1").
"""

from __future__ import annotations

from dataclasses import dataclass

from footnote.evals.golden import GoldenItem
from footnote.retrieve.pipeline import Retriever


def matches(retrieved_id: str, gold_id: str) -> bool:
    return retrieved_id == gold_id or retrieved_id.startswith(gold_id + ":")


@dataclass
class QueryResult:
    qid: str
    source: str
    gold: list[str]
    retrieved: list[str]
    first_gold_rank: int | None  # 1-based
    gold_found: int


def evaluate_query(retriever: Retriever, item: GoldenItem, top_k: int = 10) -> QueryResult:
    ret = retriever.search(item.question, top_n=top_k)
    ids = [r.provision.provision_id for r in ret.results]
    first = None
    found = set()
    for rank, rid in enumerate(ids, start=1):
        for g in item.gold_articles:
            if matches(rid, g):
                found.add(g)
                if first is None:
                    first = rank
    return QueryResult(
        qid=item.qid, source=item.source, gold=item.gold_articles,
        retrieved=ids, first_gold_rank=first, gold_found=len(found),
    )


def aggregate(rows: list[QueryResult], top_k: int = 10) -> dict:
    n = len(rows)
    if n == 0:
        return {}
    recall = sum(r.gold_found / len(r.gold) for r in rows if r.gold) / n
    mrr = sum(1 / r.first_gold_rank for r in rows if r.first_gold_rank) / n
    hit5 = sum(1 for r in rows if r.first_gold_rank and r.first_gold_rank <= 5) / n
    hitk = sum(1 for r in rows if r.first_gold_rank) / n
    return {
        "n": n,
        f"recall@{top_k}": round(recall, 4),
        "mrr": round(mrr, 4),
        "gold_in_top_5": round(hit5, 4),
        f"gold_in_top_{top_k}": round(hitk, 4),
    }
