"""Deterministic clean / normalize layer for faithful OCR output (e.g. LightOnOCR).

A faithful OCR transcribes everything — journal banners, running heads, page numbers, DOIs,
references, acknowledgements — and uses whatever heading levels it saw. This layer turns that raw
page-by-page output into clean, ``section_map``-ready text: the same shape the Qwen3-VL scientific
prompt produced (flat ``## `` headings, ``<<<PAGE n>>>`` markers, junk dropped), but with **no
generative model** — every step is a deterministic rule, so the result is byte-reproducible and
**verbatim** (words are never rewritten; lines are only dropped, and headings only marked).

It replicates the editorial decisions the VLM prompt used to make:
  * drop running heads/footers (lines that repeat across pages), page/line numbers, horizontal
    rules, DOI/copyright-only lines, and front-matter banners ("OPEN ACCESS", "RESEARCH ARTICLE")
  * scrub back-matter sections — the full set the Qwen prompt dropped, by default (configurable)
  * normalize every Markdown heading to a flat ``## ``
  * **infer missing headers** — promote standalone section-vocabulary lines, and split run-in
    headers ("Introduction: Melanoma is…"), to their own ``## `` line, the way the VLM would have

The abstract is **kept** (it lives in the corpus full_text; only the downstream per-context
concats drop it). Back-matter scrubbing defaults to the same sections the Qwen prompt removed:
references, acknowledgements, funding, author contributions, competing interests, data
availability, and supplementary (file lists).
"""
from __future__ import annotations

import re
from collections import Counter

from .sections import _ANY_HEADING, _NUMPREFIX, _VOCAB, _collapse_blank_lines

# Back-matter heading categories and the patterns that name them (matched deterministically by
# heading text — these names are unambiguous). Mirrors what the Qwen scientific prompt dropped.
_BACKMATTER_PATTERNS = {
    "references":           r"references?|bibliography|works\s+cited|literature\s+cited",
    "acknowledgements":     r"acknowledge?ments?",
    "funding":              r"funding(\s+(information|statement|sources?))?|financial\s+support|grant\s+support",
    "author contributions": r"authors?\s+contributions?|credit\s+authorship\s+contributions?|author\s+information",
    "competing interests":  r"competing\s+(financial\s+)?interests?|conflicts?\s+of\s+interest|"
                            r"declaration\s+of\s+(competing\s+)?interests?|disclosures?",
    "data availability":    r"data\s+(and\s+code\s+)?availability(\s+statement)?|code\s+availability|"
                            r"availability\s+of\s+data(\s+and\s+materials?)?|accession\s+(codes?|numbers?)",
    "supplementary":        r"supp?lement\w*|supporting\s+information",
}
_BACKMATTER_RE = {
    name: re.compile(r"^\s*" + _NUMPREFIX + r"(?:" + pat + r")\b", re.I)
    for name, pat in _BACKMATTER_PATTERNS.items()
}

# Default scrub set = everything the Qwen prompt dropped (kept verbatim out of the corpus full_text).
DEFAULT_SCRUB = tuple(_BACKMATTER_PATTERNS)

_PAGE_LINE = re.compile(r"^<<<PAGE \d+>>>\s*$")

# A whole-line section heading (anchored) → an unmarked heading we should promote to '## '.
_HEADING_ONLY = re.compile(r"^\s*" + _NUMPREFIX + _VOCAB + r"\s*[:.–-]?\s*$", re.I)
# A run-in heading: "Introduction: <prose>" / "Materials and Methods. <prose>". Colon/period
# delimiter + substantial prose after guards against body lines that merely start with a section word.
_RUNIN = re.compile(
    r"^\s*(?P<head>" + _NUMPREFIX + _VOCAB + r"(?:\s+[A-Za-z][\w-]*){0,4})\s*[:.]\s+(?P<body>\S.{20,})$", re.I
)

# Content sections — once one is seen, we're out of the front matter (stop dropping banners/meta).
_CONTENT_HEAD = re.compile(
    r"^\s*" + _NUMPREFIX + r"(abstract|summary|introduction|background|materials?|methods?|"
    r"results?|findings?|discussion|conclusions?)\b", re.I
)

_RULE = re.compile(r"^\s*([-*_=]\s*){3,}\s*$")                      # --- *** ___ ===
_PAGENUM = re.compile(r"^\s*(?:page\s+)?\d{1,4}\s*(?:/\s*\d{1,4})?\s*$", re.I)
_DOI_URL = re.compile(r"(doi:\s*\S+|https?://\S+|\bwww\.\S+)", re.I)
_COPYRIGHT = re.compile(r"(©|\(c\)\s*\d{4}|copyright\b|all\s+rights\s+reserved)", re.I)
_META_FRONT = re.compile(
    r"^\s*(edited\s+by|received\b|accepted\b|published\s+(online|on|in)|"
    r"correspond(ing|ence)|to\s+whom\s+correspondence|preprint\b)", re.I
)
_BANNER = re.compile(
    r"^\s*(open\s+access|research\s+article|review(?:\s+article)?|original\s+(?:research|article|paper)|"
    r"brief\s+(?:communication|report)|perspective|editorial|commentary|correspondence|"
    r"news\s+(?:and|&)\s+views|article|letters?)\b", re.I
)
# A single ALL-CAPS token in the front matter is almost always a journal masthead (PNAS, NATURE,
# CELL, ELIFE, SCIENCE). Real titles are multi-word, so this won't eat a title.
_JOURNAL_TOKEN = re.compile(r"^[A-Z][A-Z&.]{1,9}$")


def _norm_key(line: str) -> str:
    """Normalize a line for cross-page repeat detection (digits → '#', whitespace collapsed)."""
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line.strip().lower()))


def _split_pages(doc: str):
    """Split into [(marker_or_None, [lines])] preserving the ``<<<PAGE n>>>`` markers."""
    pages, marker, cur = [], None, []
    for ln in doc.splitlines():
        if _PAGE_LINE.match(ln.strip()):
            pages.append((marker, cur))
            marker, cur = ln.strip(), []
        else:
            cur.append(ln)
    pages.append((marker, cur))
    return pages


def _furniture_keys(pages, min_repeat: int):
    """Normalized keys of short non-heading lines that recur on >= min_repeat pages (running heads)."""
    c = Counter()
    for _marker, lines in pages:
        seen = set()
        for ln in lines:
            s = ln.strip()
            if not s or _ANY_HEADING.match(ln) or len(s) > 100:
                continue
            k = _norm_key(s)
            if k and k not in seen:
                seen.add(k)
                c[k] += 1
    return {k for k, n in c.items() if n >= min_repeat}


def _url_doi_only(s: str) -> bool:
    """True when the line is essentially just a URL/DOI (so dropping it won't lose prose)."""
    if not _DOI_URL.search(s):
        return False
    leftover = re.sub(r"[\s,;.()\[\]]+", "", _DOI_URL.sub("", s))
    return len(leftover) < 12


def _normalize_and_infer(pages, *, infer_headers, drop_furniture, drop_banners, furn_keys):
    """Per-line pass: drop junk, normalize headings to '## ', infer missing headers. Returns lines."""
    out, seen_section = [], False

    def _emit_heading(text):
        nonlocal seen_section
        out.append("## " + text)
        if _CONTENT_HEAD.match(text):
            seen_section = True

    for marker, lines in pages:
        if marker:
            out.append("")
            out.append(marker)
        for ln in lines:
            s = ln.strip()
            if not s:
                out.append("")
                continue
            if drop_furniture and _norm_key(s) in furn_keys:
                continue
            if _RULE.match(s) or _PAGENUM.match(s):
                continue

            m = _ANY_HEADING.match(ln)
            if m:
                htext = m.group(2).strip()
                if drop_banners and not seen_section and (_BANNER.match(htext) or _JOURNAL_TOKEN.match(htext)):
                    continue
                _emit_heading(htext)
                continue

            # plain line — junk filters first
            if _url_doi_only(s):
                continue
            if len(s) < 100 and _COPYRIGHT.search(s):
                continue
            if not seen_section and _META_FRONT.match(s):
                continue

            # infer missing headers
            if infer_headers and _HEADING_ONLY.match(s):
                _emit_heading(re.sub(r"[\s:.–-]+$", "", s))
                continue
            if infer_headers:
                rm = _RUNIN.match(s)
                if rm:
                    _emit_heading(rm.group("head").strip())
                    out.append(rm.group("body").strip())
                    continue

            out.append(ln)
    return out


def _classify_backmatter(htext: str) -> str:
    """Classify a heading into a back-matter category name, or 'other' (deterministic, by name)."""
    for name, rx in _BACKMATTER_RE.items():
        if rx.match(htext):
            return name
    return "other"


def _scrub_backmatter(lines, scrub_sections):
    """Drop the segment under any '## ' heading classified into scrub_sections, until the next
    non-scrub heading. Back-matter headings are matched deterministically by name."""
    scrub = {s.lower() for s in scrub_sections}
    out, scrubbing = [], False
    for ln in lines:
        if ln.startswith("## "):
            scrubbing = _classify_backmatter(ln[3:].strip()) in scrub
            if scrubbing:
                continue
        if scrubbing:
            continue
        out.append(ln)
    return out


def clean_document(doc: str, *, scrub_sections=DEFAULT_SCRUB, infer_headers: bool = True,
                   drop_furniture: bool = True, drop_banners: bool = True, min_repeat: int = 3) -> str:
    """Clean + normalize raw OCR output into ``section_map``-ready, verbatim text.

    scrub_sections: back-matter category names to drop. Default = everything the Qwen prompt
        removed: references, acknowledgements, funding, author contributions, competing interests,
        data availability, supplementary. Pass a subset (e.g. ("references", "acknowledgements"))
        to keep the rest, or () to keep all back matter. (Note: 'supplementary' also matches
        "Supplementary Methods" — drop it from scrub_sections when cleaning an SI document whose
        methods you want to keep.)
    infer_headers: promote unmarked section-vocab lines + split run-in headers to '## '.
    drop_furniture: remove running heads/footers (lines repeating on >= min_repeat pages).
    drop_banners: remove front-matter publication banners ("OPEN ACCESS", "RESEARCH ARTICLE", …).
    """
    pages = _split_pages(doc)
    furn_keys = _furniture_keys(pages, min_repeat) if drop_furniture else set()
    lines = _normalize_and_infer(
        pages, infer_headers=infer_headers, drop_furniture=drop_furniture,
        drop_banners=drop_banners, furn_keys=furn_keys,
    )
    if scrub_sections:
        lines = _scrub_backmatter(lines, scrub_sections)
    return _collapse_blank_lines("\n".join(lines))
