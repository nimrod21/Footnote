"""Chunking = selecting which provision level gets indexed.

The parser stores every level (article, paragraph, point, recital, annex, item).
A chunk strategy picks the retrieval units; finer levels remain available for
exact citation lookup regardless of strategy.
"""

from __future__ import annotations

from footnote.models import Provision


def select_chunks(provisions: list[Provision], strategy: str) -> list[Provision]:
    if strategy == "provision":
        return _provision_level(provisions)
    if strategy == "article":
        return _article_level(provisions)
    if strategy == "window":
        return _windows(provisions)
    raise ValueError(f"unknown chunk strategy: {strategy}")


def _provision_level(ps: list[Provision]) -> list[Provision]:
    """Paragraphs (points inlined), recitals, definitions, annex items.

    The natural retrieval grain: big enough to carry context, small enough
    that a hit pinpoints the provision.
    """
    out = []
    paragraphed_articles = {
        (p.instrument, p.number) for p in ps if p.type == "article" and p.paragraph
    }
    for p in ps:
        if p.type == "recital" or p.type == "definition":
            out.append(p)
        elif p.type == "article":
            if p.point is not None:
                continue  # points are inlined in their paragraph text
            if p.paragraph is not None:
                out.append(p)
            elif (p.instrument, p.number) not in paragraphed_articles:
                out.append(p)  # article with no numbered paragraphs
        elif p.type == "annex":
            if p.paragraph is not None and p.point is None:
                out.append(p)  # numbered items, sub-points inlined
    return out


def _article_level(ps: list[Provision]) -> list[Provision]:
    """Whole articles, recitals, whole annexes — the coarse control."""
    return [
        p
        for p in ps
        if (p.type in ("article", "annex") and p.paragraph is None and p.point is None)
        or p.type == "recital"
    ]


def _windows(ps: list[Provision], size: int = 512, overlap: int = 64) -> list[Provision]:
    """Naive fixed-token windows over article text — the baseline to beat."""
    out = []
    for p in _article_level(ps):
        words = p.text.split()
        step = max(1, int(size * 0.75) - int(overlap * 0.75))  # ~tokens -> words
        width = int(size * 0.75)
        for i, start in enumerate(range(0, max(1, len(words) - overlap), step)):
            piece = " ".join(words[start : start + width])
            if not piece:
                break
            out.append(
                p.model_copy(
                    update={
                        "provision_id": f"{p.provision_id}:w{i}",
                        "text": piece,
                    }
                )
            )
            if start + width >= len(words):
                break
    return out


def estimate_tokens(ps: list[Provision]) -> int:
    """Rough embedding-quota projection: ~4 chars per token."""
    return sum(len(p.text) for p in ps) // 4
