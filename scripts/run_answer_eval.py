import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from footnote.answer.generate import Answerer
from footnote.config import RunConfig
from footnote.evals.answering import aggregate_answerable, aggregate_negatives, run_suites, to_dicts
from footnote.evals.golden import load_file_sets
from footnote.evals.harness import _git_sha

cfg = RunConfig()
answerer = Answerer(cfg)
items = load_file_sets()
print(f"running {len(items)} questions...", flush=True)
rows = run_suites(answerer, items)
answerable = [r for r in rows if r.source == "authored"]
negatives = [r for r in rows if r.source.startswith("negative")]
result = {
    "suite": "answering+refusal", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "git_sha": _git_sha(), "config": cfg.model_dump(), "config_hash": cfg.config_hash(),
    "answerable": aggregate_answerable(answerable),
    "negatives": aggregate_negatives(negatives),
    "rows": to_dicts(rows),
}
Path("results/answering_baseline.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
errs = [r for r in rows if r.error]
print("errors:", len(errs), [e.qid for e in errs][:5])
print(json.dumps({"answerable": result["answerable"], "negatives": result["negatives"]}, indent=2))
