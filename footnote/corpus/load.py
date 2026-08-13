"""Load parsed provisions from the cached Official Journal files."""

from __future__ import annotations

from functools import lru_cache

from footnote.corpus.chunkers import select_chunks
from footnote.corpus.fetch import fetch_instrument
from footnote.corpus.parser import parse_instrument
from footnote.corpus.registry import INSTRUMENTS
from footnote.models import Provision


@lru_cache(maxsize=None)
def _provisions_tuple(corpus: tuple[str, ...]) -> tuple[Provision, ...]:
    out: list[Provision] = []
    for iid in corpus:
        out.extend(parse_instrument(iid, fetch_instrument(iid)))
    return tuple(out)


def load_provisions(corpus: tuple[str, ...] | None = None) -> dict[str, Provision]:
    """All provisions (every level) keyed by provision_id — the citation lookup."""
    corpus = corpus or tuple(INSTRUMENTS)
    return {p.provision_id: p for p in _provisions_tuple(corpus)}


def load_chunks(strategy: str, corpus: tuple[str, ...] | None = None) -> list[Provision]:
    """The retrieval units for a chunk strategy."""
    corpus = corpus or tuple(INSTRUMENTS)
    return select_chunks(list(_provisions_tuple(corpus)), strategy)
