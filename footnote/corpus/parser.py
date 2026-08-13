"""Parse Official Journal XHTML into Provision records.

The markup carries the legal structure as anchors (verified against both
regulations):

    <div class="eli-subdivision" id="art_6">      article
      <p class="oj-ti-art">Article 6</p>
      <div class="eli-title" id="art_6.tit_1">    heading
      <div id="006.001">                          paragraph = Art 6(1)
        points (a), (b)... as two-cell table rows
    <div class="eli-subdivision" id="rct_47">     recital
    <div class="eli-container" id="anx_III">      annex (AI Act)
    <div id="cpt_II"> / id="cpt_III.sct_2"        chapter / section

Parsing is a transcription of that structure, not heuristics.
"""

from __future__ import annotations

import re
from pathlib import Path

from lxml import etree

from footnote.corpus.registry import DEFINITION_ARTICLES, INSTRUMENTS, SHORT_NAME
from footnote.models import Provision

_WS = re.compile(r"\s+")
_PARA_ID = re.compile(r"^(\d{3})\.(\d{3})$")
_POINT_LABEL = re.compile(r"^\(([a-z]{1,3}|[ivxl]{1,5}|\d{1,2})\)$")
_LEADING_NUM = re.compile(r"^(\d{1,2})\.\s+")
_ANNEX_ITEM = re.compile(r"^(\d{1,2})\.\s+")


def _text(el: etree._Element) -> str:
    """Flatten an element to normalised text."""
    return _WS.sub(" ", " ".join(el.itertext())).strip()


def _strip_ns(root: etree._Element) -> None:
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]


class InstrumentParser:
    def __init__(self, instrument_id: str, xhtml_path: Path):
        self.iid = instrument_id
        self.short = SHORT_NAME[instrument_id]
        self.url = INSTRUMENTS[instrument_id].official_url
        root = etree.fromstring(
            xhtml_path.read_bytes(), etree.XMLParser(recover=True, huge_tree=True)
        )
        _strip_ns(root)
        self.root = root
        self.provisions: list[Provision] = []

    # -- helpers -------------------------------------------------------------

    def _make(self, **kw) -> Provision:
        p = Provision(instrument=self.iid, **kw)
        self.provisions.append(p)
        return p

    def _chapter_of(self, el: etree._Element) -> tuple[str | None, str | None]:
        chapter = section = None
        for anc in el.iterancestors("div"):
            aid = anc.get("id") or ""
            if m := re.match(r"^cpt_([IVXLC]+)\.sct_(\d+)$", aid):
                chapter, section = m.group(1), m.group(2)
            elif m := re.match(r"^cpt_([IVXLC]+)$", aid):
                chapter = chapter or m.group(1)
        return chapter, section

    def _anchor_url(self, anchor: str) -> str:
        return f"{self.url}#{anchor}"

    # -- recitals ------------------------------------------------------------

    def parse_recitals(self) -> None:
        for div in self.root.iter("div"):
            aid = div.get("id") or ""
            m = re.match(r"^rct_(\d+)$", aid)
            if not m:
                continue
            n = m.group(1)
            # recital body: table with (n) in first cell, text in second
            cells = div.findall(".//td")
            body = " ".join(_text(c) for c in cells[1:]) if len(cells) >= 2 else _text(div)
            self._make(
                provision_id=f"{self.iid}:rec:{n}",
                type="recital",
                number=n,
                text=body,
                citation_label=f"Recital {n} {self.short}",
                eurlex_url=self._anchor_url(aid),
            )

    # -- articles ------------------------------------------------------------

    def parse_articles(self) -> None:
        for div in self.root.iter("div"):
            aid = div.get("id") or ""
            if not re.match(r"^art_(\d+)$", aid):
                continue
            self._parse_article(div, aid.split("_")[1])

    def _parse_article(self, div: etree._Element, num: str) -> None:
        chapter, section = self._chapter_of(div)
        heading = None
        title_div = div.find("./div[@class='eli-title']")
        if title_div is not None:
            heading = _text(title_div)

        is_definitions = DEFINITION_ARTICLES.get(self.iid) == num

        # Article-level provision: full text
        self._make(
            provision_id=f"{self.iid}:art:{num}",
            type="article",
            number=num,
            chapter=chapter,
            section=section,
            heading=heading,
            text=_text(div),
            citation_label=f"Article {num} {self.short}",
            eurlex_url=self._anchor_url(f"art_{num}"),
        )

        # Paragraph-level: <div id="NNN.NNN">
        for sub in div.findall("./div"):
            sid = sub.get("id") or ""
            m = _PARA_ID.match(sid)
            if not m:
                continue
            para_no = str(int(m.group(2)))
            para_text = _text(sub)
            self._make(
                provision_id=f"{self.iid}:art:{num}:{para_no}",
                type="article",
                number=num,
                paragraph=para_no,
                chapter=chapter,
                section=section,
                heading=heading,
                text=para_text,
                citation_label=f"Article {num}({para_no}) {self.short}",
                eurlex_url=self._anchor_url(sid),
            )
            self._parse_points(sub, num, para_no, chapter, section, heading, is_definitions)

        # Articles with points but no numbered paragraphs (e.g. definition lists
        # rendered as a single unnumbered block) — attach points to the article.
        if not any(_PARA_ID.match(s.get("id") or "") for s in div.findall("./div")):
            self._parse_points(div, num, None, chapter, section, heading, is_definitions)

    def _parse_points(
        self,
        scope: etree._Element,
        num: str,
        para_no: str | None,
        chapter: str | None,
        section: str | None,
        heading: str | None,
        is_definitions: bool,
    ) -> None:
        for row in scope.findall(".//tr"):
            cells = row.findall("./td")
            if len(cells) != 2:
                continue
            label = _text(cells[0])
            m = _POINT_LABEL.match(label)
            if not m:
                continue
            point = m.group(1)
            body = _text(cells[1])
            if not body:
                continue
            para_part = f":{para_no}" if para_no else ""
            label_para = f"({para_no})" if para_no else ""
            self._make(
                provision_id=f"{self.iid}:art:{num}{para_part}:{point}",
                type="definition" if is_definitions else "article",
                number=num,
                paragraph=para_no,
                point=point,
                chapter=chapter,
                section=section,
                heading=heading,
                text=body,
                citation_label=f"Article {num}{label_para}({point}) {self.short}",
                eurlex_url=self._anchor_url(f"art_{num}"),
            )

    # -- annexes -------------------------------------------------------------

    def parse_annexes(self) -> None:
        for div in self.root.iter("div"):
            aid = div.get("id") or ""
            m = re.match(r"^anx_([IVXLC]+)$", aid)
            if not m:
                continue
            roman = m.group(1)
            self._make(
                provision_id=f"{self.iid}:anx:{roman}",
                type="annex",
                number=roman,
                text=_text(div),
                citation_label=f"Annex {roman} {self.short}",
                eurlex_url=self._anchor_url(aid),
            )
            # Numbered items are table rows: label cell "N." + body cell,
            # with lettered sub-points as nested two-cell rows inside the body.
            seen: set[str] = set()
            for row in div.findall(".//tr"):
                cells = row.findall("./td")
                if len(cells) != 2:
                    continue
                label = _text(cells[0])
                im = re.match(r"^(\d{1,2})\.$", label)
                if not im:
                    continue
                item = im.group(1)
                pid = f"{self.iid}:anx:{roman}:{item}"
                if pid in seen:
                    continue
                seen.add(pid)
                self._make(
                    provision_id=pid,
                    type="annex",
                    number=roman,
                    paragraph=item,
                    text=_text(cells[1]),
                    citation_label=f"Annex {roman}({item}) {self.short}",
                    eurlex_url=self._anchor_url(aid),
                )
                for sub in cells[1].findall(".//tr"):
                    scells = sub.findall("./td")
                    if len(scells) != 2:
                        continue
                    sm = _POINT_LABEL.match(_text(scells[0]))
                    if not sm or not _text(scells[1]):
                        continue
                    self._make(
                        provision_id=f"{pid}:{sm.group(1)}",
                        type="annex",
                        number=roman,
                        paragraph=item,
                        point=sm.group(1),
                        text=_text(scells[1]),
                        citation_label=f"Annex {roman}({item})({sm.group(1)}) {self.short}",
                        eurlex_url=self._anchor_url(aid),
                    )

    # -- cross-references ----------------------------------------------------

    _XREF = re.compile(r"\bArticles?\s+(\d{1,3})|\bAnnex\s+([IVXLC]+)\b")

    def resolve_cross_refs(self) -> None:
        known = {p.provision_id for p in self.provisions}
        for p in self.provisions:
            refs: list[str] = []
            for m in self._XREF.finditer(p.text):
                if m.group(1):
                    rid = f"{self.iid}:art:{m.group(1)}"
                elif m.group(2):
                    rid = f"{self.iid}:anx:{m.group(2)}"
                else:  # pragma: no cover
                    continue
                if rid in known and rid != p.provision_id and rid not in refs:
                    # don't self-reference the article a paragraph belongs to
                    if not p.provision_id.startswith(rid + ":") and rid != _parent_art(p):
                        refs.append(rid)
            p.cross_refs = refs


def _parent_art(p: Provision) -> str:
    return f"{p.instrument}:art:{p.number}"


def parse_instrument(instrument_id: str, xhtml_path: Path) -> list[Provision]:
    ip = InstrumentParser(instrument_id, xhtml_path)
    ip.parse_recitals()
    ip.parse_articles()
    ip.parse_annexes()
    ip.resolve_cross_refs()
    return ip.provisions
