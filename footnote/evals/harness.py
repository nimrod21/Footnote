"""Eval harness. A results file must contain everything needed to re-run itself:
the full RunConfig, every metric, per-query rows, quota spent, git sha, timestamp.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from footnote.config import RunConfig
from footnote.evals.golden import GoldenItem, build_multihop, build_recital_map, load_file_sets
from footnote.evals.retrieval import aggregate, evaluate_query
from footnote.retrieve.pipeline import Retriever

RECITAL_MAP_CAP = 60  # deterministic sample; full set would burn rerank quota per ablation


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd="."
        ).stdout.strip()
    except OSError:  # pragma: no cover
        return "unknown"


def answerable_items() -> list[GoldenItem]:
    rm = sorted(build_recital_map(), key=lambda i: i.qid)[:RECITAL_MAP_CAP]
    authored = [i for i in load_file_sets() if i.expect == "answered"]
    return rm + authored


def run_retrieval_suite(config: RunConfig, out_path: str | None = None) -> dict:
    retriever = Retriever(config)
    items = answerable_items()
    t0 = time.time()
    rows = [evaluate_query(retriever, item) for item in items]

    by_source: dict[str, list] = {}
    for r in rows:
        by_source.setdefault(r.source, []).append(r)

    result = {
        "suite": "retrieval",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": _git_sha(),
        "config": config.model_dump(),
        "config_hash": config.config_hash(),
        "wall_seconds": round(time.time() - t0, 1),
        "aggregate": aggregate(rows),
        "by_source": {src: aggregate(rs) for src, rs in sorted(by_source.items())},
        "rows": [asdict(r) for r in rows],
    }
    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


HEADLINE_KEYS = ["recall@10", "mrr", "gold_in_top_5"]


def compare(current: dict, baseline: dict, max_drop: float = 0.03) -> list[str]:
    """Regression check: which headline metrics dropped more than max_drop?"""
    failures = []
    for key in HEADLINE_KEYS:
        cur = current["aggregate"].get(key)
        base = baseline["aggregate"].get(key)
        if cur is not None and base is not None and cur < base - max_drop:
            failures.append(f"{key}: {base} -> {cur}")
    return failures
