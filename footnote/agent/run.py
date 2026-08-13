"""Glue: run the research loop, then synthesise the final grounded answer
from whatever the agent gathered — same verifier, same gates as single-shot."""

from __future__ import annotations

from dataclasses import dataclass

from footnote.agent.loop import AgentLoop, AgentTrace
from footnote.agent.policy import LLMPolicy
from footnote.answer.generate import Answerer, AskResult
from footnote.config import RunConfig
from footnote.models import Answer
from footnote.retrieve.pipeline import Retrieval, RetrievalResult


@dataclass
class AgentResult:
    ask: AskResult
    trace: AgentTrace
    policy_tokens: int


def ask_with_agent(question: str, config: RunConfig | None = None,
                   answerer: Answerer | None = None) -> AgentResult:
    cfg = config or RunConfig()
    answerer = answerer or Answerer(cfg)
    policy = LLMPolicy(cfg)
    loop = AgentLoop(
        answerer.retriever, policy,
        max_hops=cfg.max_hops, max_cost_usd=cfg.max_cost_usd,
    )
    gathered, trace = loop.run(question, cost_of_decision=lambda: policy.cost_usd)

    if trace.terminated_by == "refuse" or not gathered:
        reason = "The agent could not gather provisions that answer this question."
        if trace.hops and trace.hops[-1].tool == "refuse":
            reason = str(trace.hops[-1].args.get("reason") or reason)
        ask = AskResult(
            answer=Answer(verdict="refused", refusal_reason=reason),
            retrieval=None, llm=None, gate_fired=None,
        )
        return AgentResult(ask=ask, trace=trace, policy_tokens=policy.tokens)

    # Synthesise through the standard answerer: same prompt, verifier, gates.
    retrieval = Retrieval(
        results=[RetrievalResult(provision=p, score=1.0, via="agent") for p in gathered.values()],
        confidence=1.0,
    )
    ask = answerer.ask(question, skip_scope_gate=True, retrieval=retrieval)
    return AgentResult(ask=ask, trace=trace, policy_tokens=policy.tokens)
