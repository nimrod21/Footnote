"""Core data model. The citation unit is legal structure, not a token window."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

ProvisionType = Literal["article", "recital", "annex", "definition"]


class Instrument(BaseModel):
    id: str  # "gdpr" | "ai_act"
    title: str
    celex: str
    official_url: str
    version_date: str


class Provision(BaseModel):
    provision_id: str  # "gdpr:art:6:1:f"
    instrument: str
    type: ProvisionType
    number: str  # "6"
    paragraph: str | None = None  # "1"
    point: str | None = None  # "f"
    chapter: str | None = None
    section: str | None = None
    heading: str | None = None  # "Lawfulness of processing"
    text: str
    citation_label: str  # "Article 6(1)(f) GDPR" — generated once, copied everywhere
    eurlex_url: str
    cross_refs: list[str] = []


class Citation(BaseModel):
    provision_id: str
    citation_label: str  # copied from the Provision record, never model-generated
    quote: str  # must appear verbatim in the provision text — verified in code
    eurlex_url: str
    instrument: str


DISCLAIMER = (
    "Informational only, not legal advice. "
    "Only the text published in the Official Journal of the EU is authentic."
)


class Answer(BaseModel):
    verdict: Literal["answered", "refused", "out_of_scope"]
    text: str | None = None
    citations: list[Citation] = []
    confidence: float = 0.0
    refusal_reason: str | None = None
    disclaimer: str = DISCLAIMER  # always present, never optional
