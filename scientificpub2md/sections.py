"""Turn raw page-by-page VLM output into a clean document — two formats.

The VLM already marks every heading and sub-heading with a flat ``## `` prefix. From that we
produce either:

  * **headers** — the simple form: page markers stripped, every heading kept flat at ``## ``.
  * **md** — a structured markdown document: the title promoted to ``# ``, canonical top-level
    sections (Abstract, Introduction, Methods, Results, Discussion, ...) kept at ``## ``, and
    everything else the VLM flagged as a heading demoted to ``### `` (a sub-heading).

The top-level/sub-heading split is purely deterministic — a vocabulary regex, no LLM — so the
output is reproducible and the package stays dependency-light.
"""
from __future__ import annotations

import re

# Page markers inserted by the extractor between pages.
_PAGE_RE = re.compile(r"<<<PAGE \d+>>>")

# Canonical top-level section vocabulary. A heading matching this stays ``## `` in markdown mode;
# any other VLM heading is treated as a sub-section and demoted to ``### ``.
_VOCAB = (
    r"(abstract|summary|introduction|background|related\s+work|materials?\s+and\s+methods?|methods?|methodology|"
    r"approach|materials?|experimental(\s+(procedures?|section|methods?|design|setup))?|star\s+methods?|"
    r"results?(\s+and\s+discussion)?|findings?|discussion|conclusions?|concluding|references?|bibliography|"
    r"acknowledge?ments?|supp?lement\w*|supporting\s+information|author\s+contributions?|"
    r"data\s+(and\s+code\s+)?availability|code\s+availability|funding|competing\s+interests|"
    r"conflicts?\s+of\s+interest|abbreviations|ethics|consent|availability\s+of\s+data)"
)
# Allow a numbered prefix with any common separator: "2 Methods", "2. Methods", "2 | METHODS", "IV. Results".
_NUMPREFIX = r"(?:(?:\d+(?:\.\d+)*|[ivxlc]+)\s*[|.):–-]?\s*)?"
_TOPLEVEL_RE = re.compile(r"^\s*" + _NUMPREFIX + _VOCAB + r"\b", re.I)

_HEADING_LINE = re.compile(r"^\s{0,3}#{2,6}\s+(.*\S)\s*$")


def _strip_page_markers(doc: str) -> str:
    return _PAGE_RE.sub("", doc)


def _collapse_blank_lines(text: str) -> str:
    """Collapse 3+ consecutive blank lines into one, trim trailing whitespace per line."""
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


_ANY_HEADING = re.compile(r"(?m)^(\s{0,3})#{1,6}[ \t]+(.*\S)[ \t]*$")


def to_headers(doc: str) -> str:
    """Simple ``## ``-headers form: drop page markers, normalize whitespace, keep headings flat.

    The Qwen3-VL prompt already emits flat ``## `` headings; this just cleans them up. For engines
    that emit mixed Markdown levels (``flatten=True``, e.g. LightOnOCR) every ``#``..``######``
    heading is collapsed to ``## `` so the "simple headers" contract holds across engines.
    """
    text = _strip_page_markers(doc)
    return _collapse_blank_lines(text)


def flatten_headings(doc: str) -> str:
    """Collapse every Markdown heading (``#``..``######``) to a flat ``## `` heading."""
    text = _ANY_HEADING.sub(lambda m: f"{m.group(1)}## {m.group(2)}", _strip_page_markers(doc))
    return _collapse_blank_lines(text)


def passthrough_markdown(doc: str) -> str:
    """Keep an engine's native Markdown as-is — just strip page markers and tidy blank lines."""
    return _collapse_blank_lines(_strip_page_markers(doc))


def _detect_title(lines):
    """Return (title, body_start_index). The title is the first substantial non-heading text line
    that appears before the first heading (skipping short/boilerplate lines). Returns ("", 0) if
    nothing convincing is found."""
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        if _HEADING_LINE.match(line):
            break  # hit the first section heading before finding a title
        # A title is a single substantial line; skip obvious affiliation/email/doi boilerplate.
        if 12 <= len(line) <= 300 and "@" not in line and not re.search(r"https?://|\bdoi\b", line, re.I):
            return line, i + 1
        break
    return "", 0


def _detect_title_heading(lines):
    """Fallback when there's no plain-text title: the VLM often marks the paper title itself with
    ``## ``. If the document's first heading is non-canonical (not Abstract/Introduction/...), treat
    it as the title. Returns (title, body_start_index) or ("", 0)."""
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        m = _HEADING_LINE.match(line)
        if m and not _TOPLEVEL_RE.match(m.group(1).strip()):
            return m.group(1).strip(), i + 1
        return "", 0  # first content is a canonical heading or plain prose -> no title heading
    return "", 0


def to_markdown(doc: str, title: str | None = None) -> str:
    """Structured markdown: ``# `` title, ``## `` canonical sections, ``### `` sub-headings."""
    text = _strip_page_markers(doc)
    lines = text.splitlines()

    out = []
    start = 0
    if title is None:
        title, start = _detect_title(lines)
        if not title:
            title, start = _detect_title_heading(lines)
    if title:
        out.append(f"# {title.lstrip('#').strip()}")
        out.append("")

    for raw in lines[start:]:
        m = _HEADING_LINE.match(raw)
        if m:
            head = m.group(1).strip()
            level = "##" if _TOPLEVEL_RE.match(head) else "###"
            out.append(f"{level} {head}")
        else:
            out.append(raw)

    return _collapse_blank_lines("\n".join(out))


def format_document(doc: str, fmt: str = "md", *, native_markdown: bool = False,
                    title: str | None = None) -> str:
    """Render the raw extractor output as 'md', 'headers', or 'clean'.

    native_markdown=False (Qwen3-VL flat-``## `` convention): 'md' restructures into
    ``# ``/``## ``/``### `` via the section vocabulary; 'headers' keeps the flat ``## ``.
    native_markdown=True (LightOnOCR already emits structured Markdown): 'md' passes the model's
    Markdown through untouched; 'headers' flattens all heading levels to ``## ``.
    'clean' (engine-independent): deterministic clean/normalize → junk-stripped, '## '-denoted,
    page-marked, verbatim, ``section_map``-ready text with missing headers inferred.
    """
    if fmt == "clean":
        from .clean import clean_document

        return clean_document(doc)
    if fmt == "md":
        return passthrough_markdown(doc) if native_markdown else to_markdown(doc, title=title)
    if fmt == "headers":
        return flatten_headings(doc) if native_markdown else to_headers(doc)
    raise ValueError(f"unknown format {fmt!r} (expected 'md', 'headers', or 'clean')")
