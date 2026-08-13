"""LLM-driven Policy for the research loop.

Decisions are structured next-action JSON, parsed and validated — chosen over
native tool-calling because free-tier models vary wildly in tool support, and
a validated JSON contract fails visibly instead of silently.
"""

from __future__ import annotations

from pathlib import Path

from footnote.agent.loop import AgentTrace, Decision
from footnote.answer.generate import _extract_json
from footnote.answer.llm import OpenRouterClient
from footnote.config import RunConfig
from footnote.models import Provision

_VALID_TOOLS = {"search", "lookup", "follow_refs", "explain", "refine_query", "answer", "refuse"}


class LLMPolicy:
    def __init__(self, config: RunConfig, client: OpenRouterClient | None = None):
        self.config = config
        self.client = client or OpenRouterClient()
        self.system = Path("prompts/agent_v1.md").read_text(encoding="utf-8")
        self.cost_usd = 0.0
        self.tokens = 0

    def decide(
        self, question: str, trace: AgentTrace, gathered: dict[str, Provision]
    ) -> Decision:
        state = self._render_state(question, trace, gathered)
        resp = self.client.chat_with_fallback(
            [self.config.generation_model, *self.config.fallback_models],
            messages=[
                {"role": "system", "content": self.system},
                {"role": "user", "content": state},
            ],
            max_tokens=2000,
            reasoning_effort="low",  # decisions are cheap; don't let thinking eat the budget
        )
        self.cost_usd += resp.cost_usd
        self.tokens += resp.prompt_tokens + resp.completion_tokens

        parsed = _extract_json(resp.text)
        if not parsed or parsed.get("tool") not in _VALID_TOOLS:
            # Models that drift into prose usually think they're done. If we have
            # provisions, synthesise from them; only refuse when we have nothing.
            if gathered:
                return Decision("answer", {},
                                reasoning="invalid decision output with provisions gathered — synthesising")
            return Decision("refuse", {"reason": "agent produced an invalid decision"},
                            reasoning=f"unparseable decision: {resp.text[:120]}")
        return Decision(
            tool=str(parsed["tool"]),
            args=dict(parsed.get("args") or {}),
            reasoning=str(parsed.get("reasoning") or ""),
        )

    def _render_state(
        self, question: str, trace: AgentTrace, gathered: dict[str, Provision]
    ) -> str:
        parts = [f"Question: {question}\n"]
        if gathered:
            parts.append("Gathered provisions:")
            for p in gathered.values():
                parts.append(f"[{p.provision_id}] {p.citation_label}"
                             + (f" — {p.heading}" if p.heading else "")
                             + f"\n{p.text[:600]}")
        else:
            parts.append("Gathered provisions: none yet.")
        if trace.hops:
            parts.append("\nPrevious steps:")
            for h in trace.hops:
                parts.append(f"{h.n}. {h.tool}({h.args}) -> {h.observation}")
        parts.append(f"\nHops used: {len(trace.hops)}/{self.config.max_hops}. Choose the next action.")
        return "\n".join(parts)
