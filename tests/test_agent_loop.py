"""Loop bounds proven with a scripted policy — no LLM, no network."""
import pytest

from footnote.agent.loop import AgentLoop, AgentTrace, Decision
from footnote.models import Provision


def prov(pid, text="text", refs=None):
    return Provision(
        provision_id=pid, instrument="gdpr", type="article",
        number=pid.split(":")[2], text=text,
        citation_label=pid, eurlex_url="https://x", cross_refs=refs or [],
    )


class FakeRetriever:
    """Just enough of Retriever for the loop: lookup + provisions dict."""

    def __init__(self):
        self.provisions = {
            "gdpr:art:6": prov("gdpr:art:6", refs=["gdpr:art:9"]),
            "gdpr:art:9": prov("gdpr:art:9"),
        }

    def lookup(self, pid):
        return self.provisions.get(pid)

    def search(self, query, instrument=None, type=None):
        raise AssertionError("not used in these tests")


class ScriptedPolicy:
    def __init__(self, decisions):
        self.decisions = list(decisions)

    def decide(self, question, trace, gathered):
        return self.decisions.pop(0) if self.decisions else Decision("lookup", {"provision_id": "missing"})


def loop(policy, **kw):
    return AgentLoop(FakeRetriever(), policy, **kw)


def test_terminates_on_answer():
    p = ScriptedPolicy([Decision("lookup", {"provision_id": "gdpr:art:6"}),
                        Decision("answer", {})])
    gathered, trace = loop(p).run("q")
    assert trace.terminated_by == "answer" and "gdpr:art:6" in gathered


def test_max_hops_respected():
    p = ScriptedPolicy([Decision("lookup", {"provision_id": f"gdpr:art:{i}"}) for i in (6, 9, 6, 9, 6)])
    _, trace = loop(p, max_hops=3).run("q")
    assert len(trace.hops) == 3 and trace.terminated_by in ("max_hops", "no_progress")


def test_no_progress_detection():
    # two consecutive lookups of nonexistent provisions -> stop
    p = ScriptedPolicy([Decision("lookup", {"provision_id": "gdpr:art:6"}),
                        Decision("lookup", {"provision_id": "nope:1"}),
                        Decision("lookup", {"provision_id": "nope:2"}),
                        Decision("lookup", {"provision_id": "gdpr:art:9"})])
    _, trace = loop(p, max_hops=10).run("q")
    assert trace.terminated_by == "no_progress" and len(trace.hops) == 3


def test_cycle_guard():
    p = ScriptedPolicy([Decision("lookup", {"provision_id": "gdpr:art:6"}),
                        Decision("lookup", {"provision_id": "gdpr:art:6"}),
                        Decision("lookup", {"provision_id": "gdpr:art:6"})])
    gathered, trace = loop(p, max_hops=10).run("q")
    assert trace.terminated_by == "no_progress"
    assert len(gathered) == 1  # revisits added nothing


def test_cost_ceiling():
    p = ScriptedPolicy([Decision("lookup", {"provision_id": "gdpr:art:6"}) for _ in range(9)])
    _, trace = loop(p, max_hops=10, max_cost_usd=0.02).run("q", cost_of_decision=lambda: 0.01)
    assert trace.terminated_by == "cost_ceiling"


def test_follow_refs():
    p = ScriptedPolicy([Decision("lookup", {"provision_id": "gdpr:art:6"}),
                        Decision("follow_refs", {"provision_id": "gdpr:art:6"}),
                        Decision("answer", {})])
    gathered, trace = loop(p).run("q")
    assert "gdpr:art:9" in gathered and trace.terminated_by == "answer"
