"""Agent vs single-shot on multi-hop questions — does the loop earn its cost?

Ten cross-reference chains (article defers to annex). Gold coverage = how many
provisions of the chain end up cited. Single-shot sees only what one retrieval
returns; the agent can walk the references.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from footnote.agent.run import ask_with_agent
from footnote.answer.generate import Answerer
from footnote.config import RunConfig
from footnote.evals.golden import build_multihop
from footnote.evals.retrieval import matches
from footnote.evals.harness import _git_sha

N_ITEMS = 10


def score(citations, gold):
    hit = sum(1 for g in gold if any(matches(c.provision_id, g) for c in citations))
    return hit / len(gold) if gold else 0.0


def main():
    cfg = RunConfig()
    answerer = Answerer(cfg)
    items = sorted(build_multihop(), key=lambda i: i.qid)[:N_ITEMS]
    rows = []
    for it in items:
        row = {"qid": it.qid, "question": it.question, "gold": it.gold_articles}
        # single-shot
        t0 = time.time()
        try:
            r1 = answerer.ask(it.question, skip_scope_gate=True)
            row["single"] = {
                "verdict": r1.answer.verdict,
                "gold_coverage": score(r1.answer.citations, it.gold_articles),
                "n_citations": len(r1.answer.citations),
                "wall_s": round(time.time() - t0, 1),
            }
        except RuntimeError as e:
            row["single"] = {"error": str(e)[:150]}
        time.sleep(1)
        # agent
        t0 = time.time()
        try:
            ar = ask_with_agent(it.question, cfg, answerer=answerer)
            row["agent"] = {
                "verdict": ar.ask.answer.verdict,
                "gold_coverage": score(ar.ask.answer.citations, it.gold_articles),
                "n_citations": len(ar.ask.answer.citations),
                "hops": len(ar.trace.hops),
                "terminated_by": ar.trace.terminated_by,
                "policy_tokens": ar.policy_tokens,
                "wall_s": round(time.time() - t0, 1),
            }
        except RuntimeError as e:
            row["agent"] = {"error": str(e)[:150]}
        rows.append(row)
        print(f"{it.qid}: single={row['single'].get('gold_coverage')} "
              f"agent={row['agent'].get('gold_coverage')} "
              f"hops={row['agent'].get('hops')}", flush=True)
        time.sleep(1)

    def agg(mode):
        ok = [r[mode] for r in rows if "error" not in r[mode]]
        n = len(ok)
        return {
            "n": n,
            "mean_gold_coverage": round(sum(x["gold_coverage"] for x in ok) / n, 4) if n else 0,
            "answered_rate": round(sum(x["verdict"] == "answered" for x in ok) / n, 4) if n else 0,
            "mean_wall_s": round(sum(x["wall_s"] for x in ok) / n, 1) if n else 0,
        }

    result = {
        "suite": "agent_vs_single", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": _git_sha(), "config": cfg.model_dump(), "config_hash": cfg.config_hash(),
        "single": agg("single"), "agent": agg("agent"), "rows": rows,
    }
    Path("results/agent_vs_single.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"single": result["single"], "agent": result["agent"]}, indent=2))


if __name__ == "__main__":
    main()
