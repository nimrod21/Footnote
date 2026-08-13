"""Bounded legal-research loop: PLAN -> ACT -> OBSERVE, then answer or refuse.

The decision step is pluggable (an LLM in production, a scripted policy in
tests), so the bounds — max hops, cost ceiling, no-progress, cycle guard —
are provable without a network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from footnote.models import Provision
from footnote.retrieve.pipeline import Retriever


@dataclass
class Hop:
    n: int
    tool: str
    args: dict
    reasoning: str
    observation: str  # summary of what came back
    provision_ids: list[str]
    confidence: float
    cost_usd: float


@dataclass
class AgentTrace:
    hops: list[Hop] = field(default_factory=list)
    terminated_by: str = ""  # answer | refuse | max_hops | no_progress | cost_ceiling
    total_cost_usd: float = 0.0


@dataclass
class Decision:
    """What the policy wants to do next."""

    tool: str  # search | lookup | follow_refs | explain | refine_query | answer | refuse
    args: dict
    reasoning: str = ""


class Policy(Protocol):
    def decide(self, question: str, trace: AgentTrace, gathered: dict[str, Provision]) -> Decision: ...


class AgentLoop:
    def __init__(
        self,
        retriever: Retriever,
        policy: Policy,
        max_hops: int = 4,
        max_cost_usd: float = 0.05,
        min_new_confidence: float = 0.30,  # a hop must add a provision above this to count as progress
    ):
        self.retriever = retriever
        self.policy = policy
        self.max_hops = max_hops
        self.max_cost_usd = max_cost_usd
        self.min_new_confidence = min_new_confidence

    def run(self, question: str, cost_of_decision: Callable[[], float] = lambda: 0.0):
        """Returns (gathered provisions, trace). The caller turns gathered into an Answer."""
        trace = AgentTrace()
        gathered: dict[str, Provision] = {}
        visited: set[str] = set()  # cycle guard over lookup/follow targets
        stale_hops = 0

        for n in range(1, self.max_hops + 1):
            if trace.total_cost_usd >= self.max_cost_usd:
                trace.terminated_by = "cost_ceiling"
                break

            decision = self.policy.decide(question, trace, gathered)
            trace.total_cost_usd += cost_of_decision()

            if decision.tool in ("answer", "refuse"):
                trace.terminated_by = decision.tool
                trace.hops.append(Hop(
                    n=n, tool=decision.tool, args=decision.args,
                    reasoning=decision.reasoning, observation="terminal",
                    provision_ids=list(gathered), confidence=0.0,
                    cost_usd=trace.total_cost_usd,
                ))
                break

            new_provisions, observation, confidence = self._execute(decision, visited)
            fresh = [p for p in new_provisions if p.provision_id not in gathered]
            for p in fresh:
                gathered[p.provision_id] = p

            trace.hops.append(Hop(
                n=n, tool=decision.tool, args=decision.args,
                reasoning=decision.reasoning, observation=observation,
                provision_ids=[p.provision_id for p in fresh],
                confidence=confidence, cost_usd=trace.total_cost_usd,
            ))

            # No-progress: two consecutive hops adding nothing above threshold
            progressed = bool(fresh) and confidence >= self.min_new_confidence
            stale_hops = 0 if progressed else stale_hops + 1
            if stale_hops >= 2:
                trace.terminated_by = "no_progress"
                break
        else:
            trace.terminated_by = "max_hops"

        return gathered, trace

    # -- tool execution ------------------------------------------------------

    def _execute(self, d: Decision, visited: set[str]):
        r = self.retriever
        if d.tool in ("search", "refine_query"):
            ret = r.search(
                str(d.args.get("query", "")),
                instrument=d.args.get("instrument"),
                type=d.args.get("type"),
            )
            provisions = [x.provision for x in ret.results]
            return provisions, f"{len(provisions)} provisions, top confidence {ret.confidence:.2f}", ret.confidence

        if d.tool == "lookup":
            pid = str(d.args.get("provision_id", ""))
            if pid in visited:
                return [], f"{pid} already visited", 0.0
            visited.add(pid)
            p = r.lookup(pid)
            return ([p], f"found {p.citation_label}", 1.0) if p else ([], f"{pid} does not exist", 0.0)

        if d.tool == "follow_refs":
            pid = str(d.args.get("provision_id", ""))
            src = r.lookup(pid)
            if src is None:
                return [], f"{pid} does not exist", 0.0
            out = []
            for ref in src.cross_refs:
                if ref not in visited:
                    visited.add(ref)
                    if p := r.lookup(ref):
                        out.append(p)
            return out, f"{len(out)} cross-referenced provisions from {src.citation_label}", 1.0 if out else 0.0

        if d.tool == "explain":
            pid = str(d.args.get("provision_id", ""))
            src = r.lookup(pid)
            if src is None or src.type == "recital":
                return [], "nothing to explain", 0.0
            # recitals that reference this article number, cheap heuristic scan
            token = f"Article {src.number}"
            out = [
                p for p in r.provisions.values()
                if p.instrument == src.instrument and p.type == "recital"
                and token in p.text and p.provision_id not in visited
            ][:3]
            for p in out:
                visited.add(p.provision_id)
            return out, f"{len(out)} recitals referencing {src.citation_label}", 1.0 if out else 0.0

        return [], f"unknown tool {d.tool}", 0.0
