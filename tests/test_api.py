"""API surface without LLM calls: search, provisions, instruments, rate limit."""
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(
    not Path("data/raw/32016R0679.xhtml").exists(), reason="corpus not fetched")


@pytest.fixture(scope="module")
def client():
    from app.main import app
    return TestClient(app)


def test_healthz(client):
    assert client.get("/healthz").json() == {"ok": True}


def test_search_endpoint(client):
    r = client.post("/search", json={"query": "data breach notification", "instrument": "gdpr", "top_k": 5})
    assert r.status_code == 200
    hits = r.json()
    assert len(hits) == 5 and all(h["instrument"] == "gdpr" for h in hits)
    assert any("33" == h["provision_id"].split(":")[2] for h in hits)


def test_provision_endpoint(client):
    r = client.get("/provisions/gdpr:art:6:1:f")
    assert r.status_code == 200
    assert r.json()["citation_label"] == "Article 6(1)(f) GDPR"
    assert client.get("/provisions/gdpr:art:999").status_code == 404


def test_instruments(client):
    r = client.get("/instruments")
    assert {i["id"] for i in r.json()} == {"gdpr", "ai_act"}
    assert all("disclaimer" in i for i in r.json())


def test_bad_input_rejected(client):
    assert client.post("/query", json={"question": "hi"}).status_code == 422
    assert client.post("/search", json={"query": "x", "type": "bogus"}).status_code == 422


def test_openapi_has_contract(client):
    spec = client.get("/openapi.json").json()
    assert set(spec["paths"]) >= {"/query", "/search", "/provisions/{provision_id}",
                                  "/instruments", "/traces/{trace_id}", "/healthz"}
