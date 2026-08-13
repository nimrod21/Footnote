"""MCP dispatch — search/lookup/list run without any LLM."""
import pytest
from pathlib import Path

pytestmark = pytest.mark.skipif(
    not Path("data/raw/32016R0679.xhtml").exists(), reason="corpus not fetched")


def test_lookup_by_id_and_citation():
    from footnote.mcp.server import _dispatch
    r = _dispatch("lookup", {"reference": "gdpr:art:6:1:f"})
    assert r["found"] and r["citation_label"] == "Article 6(1)(f) GDPR"
    r2 = _dispatch("lookup", {"reference": "Article 6(1)(f) GDPR"})
    assert r2["found"] and r2["provision_id"] == "gdpr:art:6:1:f"


def test_lookup_trap_provision():
    from footnote.mcp.server import _dispatch
    r = _dispatch("lookup", {"reference": "Article 150 GDPR"})
    assert r["found"] is False


def test_list_instruments():
    from footnote.mcp.server import _dispatch
    r = _dispatch("list_instruments", {})
    assert {x["id"] for x in r} == {"gdpr", "ai_act"}
    assert all(x["provision_count"] > 900 for x in r)


def test_tools_declared():
    import anyio
    from footnote.mcp.server import server
    tools = anyio.run(server.list_tools)
    assert {t.name for t in tools} == {"search", "lookup", "answer", "list_instruments"}
