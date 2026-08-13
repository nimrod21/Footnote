"""M4 model bake-off: which free OpenRouter model holds the citation contract?

Retrieval runs once per question; each candidate model answers from the same
provisions. Measured, not assumed: JSON validity, citation survival (verbatim +
real id), refusal correctness on the two questions that must be refused.

Run:  .venv/Scripts/python scripts/bakeoff_m4.py
Writes results/bakeoff_m4.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from footnote.answer.generate import Answerer
from footnote.config import RunConfig

CANDIDATES = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-31b-it:free",
    "openai/gpt-oss-20b:free",
]

# (question, should_refuse)
QUESTIONS = [
    ("When can I process personal data based on legitimate interest?", False),
    ("Is an AI system used for CV screening and recruitment considered high-risk?", False),
    ("What rights does a person have when a decision about them is made solely by automated processing?", False),
    ("Do I need to appoint a Data Protection Officer if I am a small company?", False),
    ("What are the transparency obligations for chatbots under the AI Act?", False),
    ("How long do I have to report a personal data breach to the supervisory authority?", False),
    ("What is the maximum fine for data protection violations in Georgia?", True),
    ("Does the AI Act regulate AI systems developed exclusively for military purposes?", True),
    # (military AI is excluded by Art 2(3) AI Act — the *correct* grounded answer cites the
    # exclusion; a refusal is also acceptable. Scored as: answered-with-citation OR refused.)
]


def main() -> None:
    cfg = RunConfig(generation_model=CANDIDATES[0])
    answerer = Answerer(cfg)

    print("retrieving once per question...")
    retrievals = {q: answerer.retriever.search(q) for q, _ in QUESTIONS}

    report: dict = {}
    for model in CANDIDATES:
        answerer.config.generation_model = model
        stats = {"json_ok": 0, "citations_ok": 0, "citations_rejected": 0,
                 "refusal_ok": 0, "latency_ms": [], "rows": []}
        for q, should_refuse in QUESTIONS:
            try:
                r = answerer.ask(q, skip_scope_gate=True, retrieval=retrievals[q])
            except RuntimeError as e:
                stats["rows"].append({"q": q, "error": str(e)[:200]})
                continue
            a = r.answer
            json_ok = r.gate_fired != "verification" or "not valid JSON" not in (a.refusal_reason or "")
            stats["json_ok"] += json_ok
            if r.llm:
                stats["latency_ms"].append(r.llm.latency_ms)
            if should_refuse:
                ok = a.verdict == "refused" or bool(a.verdict == "answered" and a.citations)
                stats["refusal_ok"] += int(ok)
            elif a.verdict == "answered" and a.citations:
                stats["citations_ok"] += 1
            stats["rows"].append({
                "q": q[:60], "verdict": a.verdict, "gate": r.gate_fired,
                "n_citations": len(a.citations),
                "labels": [c.citation_label for c in a.citations],
            })
            time.sleep(1)  # be polite to the free tier
        lat = stats.pop("latency_ms")
        stats["median_latency_ms"] = sorted(lat)[len(lat) // 2] if lat else None
        report[model] = stats
        print(f"{model}: json={stats['json_ok']}/8 grounded={stats['citations_ok']}/6 "
              f"refusals={stats['refusal_ok']}/2 lat={stats['median_latency_ms']}ms")

    out = Path("results/bakeoff_m4.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"written {out}")


if __name__ == "__main__":
    main()
