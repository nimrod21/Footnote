"""Single-shot grounded answering: three gates, then generation, then verification.

Gate 1 (scope): is this about EU data protection / AI regulation at all?
Gate 2 (confidence): retrieval too weak -> refuse without a generation call.
Gate 3 (verification): citations must survive verify_all, else degrade to refusal.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from footnote.config import RunConfig
from footnote.answer.llm import LLMResponse, OpenRouterClient
from footnote.answer.verify import verify_all
from footnote.models import Answer
from footnote.retrieve.pipeline import Retrieval, Retriever

PROMPT_DIR = Path("prompts")

SCOPE_PROMPT = (
    "Is the following question about EU data protection law (GDPR) or EU AI "
    "regulation (the AI Act)? Reply with exactly one word: yes or no.\n\n"
    "Question: {q}"
)


@dataclass
class AskResult:
    answer: Answer
    retrieval: Retrieval | None
    llm: LLMResponse | None
    gate_fired: str | None  # scope | confidence | verification | None
    prompt_version: str = "v1"


def _load_prompt(version: str) -> str:
    return (PROMPT_DIR / f"answer_{version}.md").read_text(encoding="utf-8")


def _extract_json(text: str) -> dict | None:
    """Parse model output as JSON, tolerating markdown fences and prose margins."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.S)  # last resort: outermost braces
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


class Answerer:
    def __init__(self, config: RunConfig | None = None, retriever: Retriever | None = None):
        self.config = config or RunConfig()
        self.retriever = retriever or Retriever(self.config)
        self.client = OpenRouterClient()

    def ask(
        self,
        question: str,
        instrument: str | None = None,
        skip_scope_gate: bool = False,
        retrieval: Retrieval | None = None,  # precomputed (bake-off / eval reuse)
    ) -> AskResult:
        cfg = self.config

        # Gate 1 — scope (one cheap call; skippable for eval harness reuse)
        if not skip_scope_gate and not self._in_scope(question):
            return AskResult(
                answer=Answer(
                    verdict="out_of_scope",
                    refusal_reason="Not a question about the GDPR or the EU AI Act.",
                ),
                retrieval=None, llm=None, gate_fired="scope",
            )

        if retrieval is None:
            retrieval = self.retriever.search(question, instrument=instrument)

        # Gate 2 — retrieval confidence
        if retrieval.confidence < cfg.confidence_threshold:
            return AskResult(
                answer=Answer(
                    verdict="refused",
                    confidence=retrieval.confidence,
                    refusal_reason=(
                        f"No sufficiently relevant provision found "
                        f"(confidence {retrieval.confidence:.2f} < {cfg.confidence_threshold:.2f})."
                    ),
                ),
                retrieval=retrieval, llm=None, gate_fired="confidence",
            )

        # Generation
        retrieved = {r.provision.provision_id: r.provision for r in retrieval.results}
        provisions_block = "\n\n".join(
            f"[{p.provision_id}] {p.citation_label}"
            + (f" — {p.heading}" if p.heading else "")
            + f"\n{p.text}"
            for p in retrieved.values()
        )
        llm = self.client.chat(
            model=cfg.generation_model,
            messages=[
                {"role": "system", "content": _load_prompt(cfg.prompt_version)},
                {"role": "user", "content": f"Provisions:\n\n{provisions_block}\n\nQuestion: {question}"},
            ],
        )
        parsed = _extract_json(llm.text)

        # Gate 3 — verification
        if parsed is None or parsed.get("verdict") not in ("answered", "refused"):
            return self._verification_refusal(retrieval, llm, "Model output was not valid JSON.")
        if parsed["verdict"] == "refused":
            return AskResult(
                answer=Answer(
                    verdict="refused",
                    confidence=retrieval.confidence,
                    refusal_reason=str(parsed.get("reason") or "The provisions do not answer this."),
                ),
                retrieval=retrieval, llm=llm, gate_fired=None,
            )
        citations, rejected = verify_all(list(parsed.get("citations") or []), retrieved)
        if not citations:
            return self._verification_refusal(
                retrieval, llm, f"No citation survived verification ({rejected} rejected)."
            )
        return AskResult(
            answer=Answer(
                verdict="answered",
                text=str(parsed.get("answer") or ""),
                citations=citations,
                confidence=retrieval.confidence,
            ),
            retrieval=retrieval, llm=llm, gate_fired=None,
        )

    def _verification_refusal(self, retrieval, llm, why: str) -> AskResult:
        return AskResult(
            answer=Answer(
                verdict="refused",
                confidence=retrieval.confidence,
                refusal_reason=f"Answer withheld: {why}",
            ),
            retrieval=retrieval, llm=llm, gate_fired="verification",
        )

    def _in_scope(self, question: str) -> bool:
        resp = self.client.chat(
            model=self.config.generation_model,
            messages=[{"role": "user", "content": SCOPE_PROMPT.format(q=question)}],
            max_tokens=1500,
        )
        return "yes" in resp.text.strip().lower()[:20]
