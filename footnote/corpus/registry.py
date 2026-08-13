"""Instrument registry.

OJ versions are used rather than consolidated CELEX ids: the consolidated XHTML
(verified 2026-08-13) drops recital and paragraph anchors, and the AI Act has no
consolidated edition yet. GDPR corrigenda are a documented limitation.
"""

from footnote.models import Instrument

INSTRUMENTS: dict[str, Instrument] = {
    "gdpr": Instrument(
        id="gdpr",
        title="Regulation (EU) 2016/679 (GDPR)",
        celex="32016R0679",
        official_url="https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679",
        version_date="2016-05-04",
    ),
    "ai_act": Instrument(
        id="ai_act",
        title="Regulation (EU) 2024/1689 (AI Act)",
        celex="32024R1689",
        official_url="https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32024R1689",
        version_date="2024-07-12",
    ),
}

# Short names used in citation labels: "Article 6(1)(f) GDPR"
SHORT_NAME = {"gdpr": "GDPR", "ai_act": "AI Act"}

# Definition articles: every point in these becomes type="definition"
DEFINITION_ARTICLES = {"gdpr": "4", "ai_act": "3"}
