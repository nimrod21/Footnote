from footnote.config import RunConfig, Secrets
from footnote.models import Answer, Citation, Provision, DISCLAIMER


def test_config_hash_stable_and_sensitive():
    a, b = RunConfig(), RunConfig()
    assert a.config_hash() == b.config_hash()
    assert a.config_hash() != RunConfig(rerank_enabled=False).config_hash()


def test_secrets_load_from_env():
    s = Secrets()
    assert s.jina_api_key and s.qdrant_url and s.openrouter_api_key


def test_answer_always_carries_disclaimer():
    a = Answer(verdict="refused", refusal_reason="no supporting provision")
    assert a.disclaimer == DISCLAIMER


def test_provision_model():
    p = Provision(
        provision_id="gdpr:art:6:1:f", instrument="gdpr", type="article",
        number="6", paragraph="1", point="f", heading="Lawfulness of processing",
        text="processing is necessary for the purposes of the legitimate interests...",
        citation_label="Article 6(1)(f) GDPR",
        eurlex_url="https://eur-lex.europa.eu/eli/reg/2016/679/oj#art_6",
    )
    assert p.citation_label == "Article 6(1)(f) GDPR"
