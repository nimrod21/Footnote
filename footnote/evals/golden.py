"""Golden set builders.

Provenance matters — each item records where its ground truth came from:
- recital_map  (mechanical): a recital that names exactly one Article becomes a
  query; the named article is gold. Nobody authored anything.
- multihop     (mechanical): cross-reference chains from the parser. A question
  whose answer requires the chain's tail is gold for the agent eval.
- authored     (hand-written): realistic questions mapped to provisions by the
  author, not verified by a lawyer — stated plainly in the README.
- negatives    (hand-written): out-of-scope, unanswerable-in-domain, and trap
  questions (provisions that don't exist).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from footnote.corpus.load import load_provisions

GOLDEN_DIR = Path("evals/golden")


@dataclass
class GoldenItem:
    qid: str
    question: str
    gold_articles: list[str]  # article-level ids, e.g. "gdpr:art:33"
    source: str  # recital_map | multihop | authored | negative:*
    expect: str = "answered"  # answered | refused | out_of_scope
    hops: list[str] = field(default_factory=list)  # gold chain for multihop items


_ONE_ARTICLE = re.compile(r"\bArticle\s+(\d{1,3})\b")


def build_recital_map() -> list[GoldenItem]:
    """Recitals naming exactly one distinct Article -> (recital text, that article)."""
    provisions = load_provisions()
    items = []
    for p in provisions.values():
        if p.type != "recital":
            continue
        arts = {m.group(1) for m in _ONE_ARTICLE.finditer(p.text)}
        if len(arts) != 1:
            continue
        art = next(iter(arts))
        target = f"{p.instrument}:art:{art}"
        if target not in provisions:
            continue
        # strip the giveaway "Article N" mention from the query text
        query = _ONE_ARTICLE.sub("the relevant provision", p.text)[:600]
        items.append(GoldenItem(
            qid=f"rm_{p.provision_id.replace(':', '_')}",
            question=query,
            gold_articles=[target],
            source="recital_map",
        ))
    return items


def build_multihop() -> list[GoldenItem]:
    """Articles that defer to an Annex or exactly one other Article.

    The chain head alone is an incomplete answer; gold includes the tail.
    """
    provisions = load_provisions()
    items = []
    for p in provisions.values():
        if p.type != "article" or p.paragraph or p.point:
            continue
        anx_refs = [r for r in p.cross_refs if ":anx:" in r]
        if not anx_refs or not p.heading:
            continue
        items.append(GoldenItem(
            qid=f"mh_{p.provision_id.replace(':', '_')}",
            question=f"What does the {p.instrument.replace('_', ' ').upper()} require regarding "
                     f"{p.heading.lower()}, including everything it lists elsewhere?",
            gold_articles=[p.provision_id] + anx_refs[:2],
            source="multihop",
            hops=[p.provision_id] + anx_refs[:2],
        ))
    return items


def load_file_sets() -> list[GoldenItem]:
    """Authored + negative sets from committed JSON files."""
    items = []
    for f in sorted(GOLDEN_DIR.glob("*.json")):
        for row in json.loads(f.read_text(encoding="utf-8")):
            items.append(GoldenItem(**row))
    return items


def full_set() -> list[GoldenItem]:
    return build_recital_map() + build_multihop() + load_file_sets()
