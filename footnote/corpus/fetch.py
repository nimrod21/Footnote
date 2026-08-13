"""Fetch regulation XHTML from the Publications Office Cellar API.

The EUR-Lex website itself is bot-blocked (HTTP 202, empty body) — do not use it.
Cellar serves the same Official Journal markup with proper content negotiation.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from footnote.corpus.registry import INSTRUMENTS

CELLAR_URL = "http://publications.europa.eu/resource/celex/{celex}"
RAW_DIR = Path("data/raw")


def fetch_instrument(instrument_id: str, force: bool = False) -> Path:
    """Download (or reuse cached) XHTML for an instrument. Returns the file path."""
    inst = INSTRUMENTS[instrument_id]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{inst.celex}.xhtml"
    if path.exists() and not force:
        return path

    resp = httpx.get(
        CELLAR_URL.format(celex=inst.celex),
        headers={"Accept": "application/xhtml+xml", "Accept-Language": "eng"},
        follow_redirects=True,
        timeout=120,
    )
    resp.raise_for_status()
    if len(resp.content) < 100_000:  # both regulations are ~1 MB; tiny = error page
        raise RuntimeError(
            f"Suspiciously small response for {inst.celex}: {len(resp.content)} bytes"
        )
    path.write_bytes(resp.content)
    return path
