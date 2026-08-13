"""Citation verification — enforced in code, never trusted from the model.

1. Cited id must be among the retrieved provisions.
2. Quote must appear verbatim in that provision's text (whitespace- and
   typography-normalised; no semantic slack).
3. citation_label is copied from the provision record, never model-generated.
"""

from __future__ import annotations

import re

from footnote.models import Citation, Provision

# The OJ text uses typographic quotes and dashes; models often normalise them.
_TYPO = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                       "–": "-", "—": "-", " ": " "})


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.translate(_TYPO)).strip().lower()


def verify_citation(
    cited_id: str, quote: str, retrieved: dict[str, Provision]
) -> Citation | None:
    """Return a verified Citation, or None if the model fabricated anything."""
    prov = retrieved.get(cited_id)
    if prov is None:
        return None
    if not quote or _norm(quote) not in _norm(prov.text):
        return None
    return Citation(
        provision_id=prov.provision_id,
        citation_label=prov.citation_label,  # from the record, not the model
        quote=quote.strip(),
        eurlex_url=prov.eurlex_url,
        instrument=prov.instrument,
    )


def verify_all(
    raw_citations: list[dict], retrieved: dict[str, Provision]
) -> tuple[list[Citation], int]:
    """Verify every claimed citation. Returns (verified, rejected_count)."""
    verified: list[Citation] = []
    rejected = 0
    for rc in raw_citations:
        c = verify_citation(str(rc.get("id", "")), str(rc.get("quote", "")), retrieved)
        if c is None:
            rejected += 1
        elif all(v.provision_id != c.provision_id or v.quote != c.quote for v in verified):
            verified.append(c)
    return verified, rejected
