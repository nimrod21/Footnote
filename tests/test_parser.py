"""Parser acid tests against the cached Official Journal files.

Skipped when data/raw is absent (fresh clone before first ingest).
"""
from pathlib import Path

import pytest

from footnote.corpus.chunkers import select_chunks
from footnote.corpus.parser import parse_instrument

GDPR = Path("data/raw/32016R0679.xhtml")
AIACT = Path("data/raw/32024R1689.xhtml")

pytestmark = pytest.mark.skipif(not GDPR.exists(), reason="corpus not fetched")


@pytest.fixture(scope="module")
def gdpr():
    return {p.provision_id: p for p in parse_instrument("gdpr", GDPR)}


@pytest.fixture(scope="module")
def ai_act():
    return {p.provision_id: p for p in parse_instrument("ai_act", AIACT)}


def test_gdpr_counts(gdpr):
    arts = [p for p in gdpr.values() if p.type == "article" and not p.paragraph and not p.point]
    recs = [p for p in gdpr.values() if p.type == "recital"]
    assert len(arts) == 99 and len(recs) == 173


def test_art_6_1_f_verbatim(gdpr):
    p = gdpr["gdpr:art:6:1:f"]
    assert p.citation_label == "Article 6(1)(f) GDPR"
    assert p.text.startswith("processing is necessary for the purposes of the legitimate interests")
    # quote-verification invariant: the point text appears inside its paragraph
    assert p.text in gdpr["gdpr:art:6:1"].text


def test_definitions_extracted(gdpr):
    defs = [p for p in gdpr.values() if p.type == "definition"]
    assert len(defs) >= 26  # Art 4 GDPR defines 26 terms
    assert any("controller" in d.text[:40] for d in defs)


def test_cross_refs(gdpr):
    assert "gdpr:art:9" in gdpr["gdpr:art:6"].cross_refs


def test_ai_act_annex_iii_item_4(ai_act):
    p = ai_act["ai_act:anx:III:4"]
    assert p.citation_label == "Annex III(4) AI Act"
    assert p.text.startswith("Employment")
    assert "recruitment" in ai_act["ai_act:anx:III:4:a"].text


def test_ai_act_art6_references_annex_iii(ai_act):
    assert "ai_act:anx:III" in ai_act["ai_act:art:6"].cross_refs


def test_chunk_strategies_differ(gdpr):
    ps = list(gdpr.values())
    prov = select_chunks(ps, "provision")
    art = select_chunks(ps, "article")
    win = select_chunks(ps, "window")
    assert len(prov) > len(art) > 0 and len(win) > 0
    # provision-level chunks never include bare points
    assert all(p.point is None or p.type == "definition" for p in prov)


def test_every_provision_has_label_and_url(gdpr, ai_act):
    for d in (gdpr, ai_act):
        for p in d.values():
            assert p.citation_label and p.eurlex_url and p.text
