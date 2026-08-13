"""Fusion maths and citation parsing — no network."""
from footnote.retrieve.fusion import rrf
from footnote.retrieve.pipeline import parse_citation


def test_rrf_maths():
    fused = rrf([["a", "b", "c"], ["b", "a"]], k=60)
    scores = dict(fused)
    assert abs(scores["a"] - (1 / 61 + 1 / 62)) < 1e-9
    assert abs(scores["b"] - (1 / 62 + 1 / 61)) < 1e-9
    assert abs(scores["c"] - 1 / 63) < 1e-9
    assert fused[-1][0] == "c"


def test_rrf_single_list():
    assert [x[0] for x in rrf([["x", "y"]])] == ["x", "y"]


def test_parse_citation_full():
    assert parse_citation("What does Article 6(1)(f) GDPR say?") == ["gdpr:art:6:1:f"]


def test_parse_citation_short_and_recital():
    assert "gdpr:art:22" in parse_citation("explain Art. 22 GDPR")
    assert "gdpr:rec:47" in parse_citation("see Recital 47 GDPR")


def test_parse_citation_annex():
    assert "ai_act:anx:III:4" in parse_citation("Annex III(4) AI Act")


def test_parse_citation_no_instrument_expands_both():
    ids = parse_citation("see Article 5")
    assert "gdpr:art:5" in ids and "ai_act:art:5" in ids


def test_no_citation_in_plain_question():
    assert parse_citation("Can I process data based on legitimate interest?") == []
