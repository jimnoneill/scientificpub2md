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
from html.parser import HTMLParser

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


# --------------------------------------------------------------------------------------
# HTML table -> Markdown table (LightOnOCR emits tables as HTML; the 'md' format prefers Markdown)
# --------------------------------------------------------------------------------------
_TABLE_BLOCK = re.compile(r"<table\b.*?</table\s*>", re.I | re.S)
# A few inline tags map to Markdown; everything else inside a cell is dropped to its text.
_INLINE_MD = {"strong": "**", "b": "**", "em": "*", "i": "*", "code": "`"}


class _TableExtractor(HTMLParser):
    """Collect an HTML ``<table>`` into rows of cell strings, mapping a few inline tags to Markdown."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []          # list[list[str]] — one list of cell texts per <tr>
        self._row = None
        self._cell = None

    def _emit(self, s):
        if self._cell is not None:
            self._cell.append(s)

    def handle_starttag(self, tag, attrs):
        t = tag.lower()
        if t == "tr":
            self._row = []
        elif t in ("td", "th"):
            self._cell = []
        elif t == "br":
            self._emit(" ")
        elif t in _INLINE_MD:
            self._emit(_INLINE_MD[t])

    def handle_endtag(self, tag):
        t = tag.lower()
        if t in ("td", "th"):
            if self._cell is not None and self._row is not None:
                cell = re.sub(r"\s+", " ", "".join(self._cell)).strip().replace("|", r"\|")
                self._row.append(cell)
            self._cell = None
        elif t == "tr":
            if self._row:
                self.rows.append(self._row)
            self._row = None
        elif t in _INLINE_MD:
            self._emit(_INLINE_MD[t])

    def handle_data(self, data):
        self._emit(data)


def _table_to_markdown(html: str) -> str:
    """Convert one ``<table>...</table>`` block to a GitHub Markdown table (first row = header).

    Returns the original HTML unchanged if it can't be parsed into a grid (e.g. nested/empty), so
    nothing is ever lost."""
    p = _TableExtractor()
    try:
        p.feed(html)
        p.close()
    except Exception:
        return html
    rows = p.rows
    if not rows:
        return html
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    header, body = rows[0], rows[1:]
    md = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * ncol) + " |"]
    md += ["| " + " | ".join(r) + " |" for r in body]
    return "\n" + "\n".join(md) + "\n"


def html_tables_to_markdown(text: str) -> str:
    """Replace every HTML ``<table>`` block in ``text`` with an equivalent Markdown table."""
    return _TABLE_BLOCK.sub(lambda m: _table_to_markdown(m.group(0)), text)


# A Markdown ATX heading at any level (1–6). Unlike _HEADING_LINE this also matches a single ``# ``.
_MD_HEADING = re.compile(r"^(\s{0,3})#{1,6}[ \t]+(.*\S)[ \t]*$")
# A fenced code block boundary (``` or ~~~) — headings inside must not be re-levelled.
_CODE_FENCE = re.compile(r"^\s{0,3}(```|~~~)")
# A bare single-integer-numbered top-level section ("3. New Items", "4 Action Items") — but NOT a
# decimal sub-section ("3.1 …"), which stays a sub-heading.
_NUM_TOPLEVEL = re.compile(r"^\s*\d{1,3}[.):]?\s+\S")
_NUM_SUBSECTION = re.compile(r"^\s*\d{1,3}\.\d")


def _is_toplevel_section(text: str) -> bool:
    """True for a heading that belongs at ``## ``: a canonical-vocabulary section or a bare
    single-integer-numbered section. A decimal sub-section ("3.1 …") is not top-level."""
    if _NUM_SUBSECTION.match(text):
        return False
    return bool(_TOPLEVEL_RE.match(text) or _NUM_TOPLEVEL.match(text))


def restructure_markdown(doc: str) -> str:
    """Re-level a native-Markdown engine's output into a consistent hierarchy.

    Pages are transcribed independently, so a native-Markdown engine (LightOnOCR) can mark the *same*
    logical section at different ``#`` levels on different pages (e.g. a numbered section coming out
    ``##`` on one page and ``#`` on another, and running heads emitted as stray ``#`` titles). This
    normalizes to the shape ``to_markdown`` produces: a single ``# `` title, canonical/numbered
    top-level sections at ``## ``, every other heading at ``### ``. HTML tables (LightOnOCR's native
    table format) are converted to Markdown tables; body text, LaTeX, and fenced code are otherwise
    left untouched — only heading markers and table syntax change, so the prose stays verbatim.
    """
    lines = html_tables_to_markdown(_strip_page_markers(doc)).splitlines()

    # Find the title: the first heading (outside code fences) that is not itself a top-level section,
    # or a substantial plain-text line before the first heading (matching _detect_title's rules).
    title_idx, in_fence = None, False
    for i, raw in enumerate(lines):
        if _CODE_FENCE.match(raw):
            in_fence = not in_fence
            continue
        if in_fence or not raw.strip():
            continue
        m = _MD_HEADING.match(raw)
        if m:
            if not _is_toplevel_section(m.group(2).strip()):
                title_idx = i
            break
        line = raw.strip()
        if 12 <= len(line) <= 300 and "@" not in line and not re.search(r"https?://|\bdoi\b", line, re.I):
            title_idx = i
        break

    out, in_fence = [], False
    for i, raw in enumerate(lines):
        if _CODE_FENCE.match(raw):
            in_fence = not in_fence
            out.append(raw)
            continue
        if in_fence:
            out.append(raw)
            continue
        if i == title_idx:
            m = _MD_HEADING.match(raw)
            out.append(f"# {(m.group(2) if m else raw).strip()}")
            continue
        m = _MD_HEADING.match(raw)
        if m:
            head = m.group(2).strip()
            out.append(f"{'##' if _is_toplevel_section(head) else '###'} {head}")
        else:
            out.append(raw)
    return _collapse_blank_lines("\n".join(out))


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
    native_markdown=True (LightOnOCR emits mixed Markdown levels): 'md' re-levels the model's
    headings into a consistent ``# ``/``## ``/``### `` hierarchy (body/tables/LaTeX untouched);
    'headers' flattens all heading levels to ``## ``.
    'clean' (engine-independent): deterministic clean/normalize → junk-stripped, '## '-denoted,
    page-marked, verbatim, parser-ready text with missing headers inferred.
    """
    if fmt == "clean":
        from .clean import clean_document

        return clean_document(doc)
    if fmt == "md":
        return restructure_markdown(doc) if native_markdown else to_markdown(doc, title=title)
    if fmt == "headers":
        return flatten_headings(doc) if native_markdown else to_headers(doc)
    raise ValueError(f"unknown format {fmt!r} (expected 'md', 'headers', or 'clean')")
