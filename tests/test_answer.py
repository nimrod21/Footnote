"""Citation verification and JSON extraction — no network."""
from footnote.answer.generate import _extract_json
from footnote.answer.verify import verify_all, verify_citation
from footnote.models import Provision


def prov(pid="gdpr:art:6:1:f", text="processing is necessary for the purposes of the legitimate interests pursued by the controller"):
    return Provision(
        provision_id=pid, instrument="gdpr", type="article", number="6",
        paragraph="1", point="f", text=text,
        citation_label="Article 6(1)(f) GDPR", eurlex_url="https://x/#art_6",
    )


RETRIEVED = {"gdpr:art:6:1:f": prov()}


def test_verbatim_quote_accepted():
    c = verify_citation("gdpr:art:6:1:f", "legitimate interests pursued by the controller", RETRIEVED)
    assert c is not None and c.citation_label == "Article 6(1)(f) GDPR"


def test_fabricated_quote_rejected():
    assert verify_citation("gdpr:art:6:1:f", "data may be processed freely", RETRIEVED) is None


def test_unretrieved_id_rejected():
    assert verify_citation("gdpr:art:99", "anything", RETRIEVED) is None


def test_typographic_normalisation():
    p = prov(text="the controller's obligations — including 'notification'")
    c = verify_citation(p.provision_id, "the controller's obligations - including 'notification'",
                        {p.provision_id: p})
    assert c is not None


def test_verify_all_counts_rejections():
    raw = [
        {"id": "gdpr:art:6:1:f", "quote": "legitimate interests pursued by the controller"},
        {"id": "gdpr:art:6:1:f", "quote": "INVENTED TEXT"},
        {"id": "nope", "quote": "x"},
    ]
    verified, rejected = verify_all(raw, RETRIEVED)
    assert len(verified) == 1 and rejected == 2


def test_extract_json_plain_and_fenced():
    assert _extract_json('{"verdict": "refused", "reason": "x"}')["verdict"] == "refused"
    assert _extract_json('```json\n{"verdict": "answered"}\n```')["verdict"] == "answered"
    assert _extract_json('Sure! Here it is:\n{"verdict": "answered", "citations": []}')["verdict"] == "answered"
    assert _extract_json("no json here") is None
