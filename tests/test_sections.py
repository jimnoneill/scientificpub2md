"""Tests for the format layer (no model needed — pure text transforms)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scientificpub2md.sections import (  # noqa: E402
    flatten_headings,
    format_document,
    passthrough_markdown,
    restructure_markdown,
    to_headers,
    to_markdown,
)

# A tiny stand-in for raw VLM output: page markers + flat '## ' headings the model emits.
RAW = """\

<<<PAGE 1>>>
A Deterministic Method for Parsing Manuscripts

## Abstract
We present a method.

## Introduction
Background prose here.

<<<PAGE 2>>>
## Materials and Methods

## Bacterial Strains and Growth Conditions
Strains were grown.

## Statistical Analysis
We used t-tests.

## Results
We found things.

## Discussion
It means something.
"""


def test_headers_strips_page_markers_and_keeps_flat():
    out = to_headers(RAW)
    assert "<<<PAGE" not in out
    assert out.count("## ") == 7  # all seven headings remain flat at '##'
    assert "### " not in out


def test_markdown_promotes_title_and_demotes_subheadings():
    out = to_markdown(RAW)
    assert "<<<PAGE" not in out
    # Title detected from the first substantial line and promoted to '#'.
    assert out.startswith("# A Deterministic Method for Parsing Manuscripts")
    # Canonical sections stay '##'.
    for sec in ("## Abstract", "## Introduction", "## Materials and Methods", "## Results", "## Discussion"):
        assert sec in out
    # Non-canonical sub-headings demoted to '###'.
    assert "### Bacterial Strains and Growth Conditions" in out
    assert "### Statistical Analysis" in out


# When the VLM marks the title itself as a '## ' heading (common), markdown mode should still
# promote it to '#' rather than demoting it to '###'.
RAW_TITLE_AS_HEADING = """\

<<<PAGE 1>>>
## Emergence and evolution of heterocyte glycolipid biosynthesis

## Abstract
Heterocytes are specialized cells.

## Results and Discussion

## Genomic Prediction of HG Biosynthesis
We searched genomes.
"""


def test_markdown_promotes_title_marked_as_heading():
    out = to_markdown(RAW_TITLE_AS_HEADING)
    assert out.startswith("# Emergence and evolution of heterocyte glycolipid biosynthesis")
    assert "### Emergence" not in out  # not demoted
    assert "## Abstract" in out
    assert "## Results and Discussion" in out
    assert "### Genomic Prediction of HG Biosynthesis" in out  # real sub-heading demoted


# A native-Markdown engine (LightOnOCR) emits mixed heading levels + tables/LaTeX directly.
RAW_NATIVE_MD = """\

<<<PAGE 1>>>
# A Faithful OCR of a Manuscript

## Methods

### Cell Culture
Cells were grown at 37C.

| Reagent | Amount |
|---|---|
| NaCl | 5 g |

The energy is $E = mc^2$.
"""


def test_native_markdown_passthrough_preserves_levels_and_tables():
    out = passthrough_markdown(RAW_NATIVE_MD)
    assert "<<<PAGE" not in out
    assert "# A Faithful OCR of a Manuscript" in out  # title level preserved
    assert "## Methods" in out
    assert "### Cell Culture" in out                   # sub-heading NOT demoted further
    assert "| Reagent | Amount |" in out               # markdown table kept
    assert "$E = mc^2$" in out                          # LaTeX kept


# Pages are OCR'd independently, so a native engine can give sibling sections different levels and
# emit running heads as stray '# ' titles — exactly what was observed on a real 4-page document.
RAW_INCONSISTENT_MD = """\

<<<PAGE 1>>>
# Weekly Meeting Update

## 1. Action Items

## 2. Milestone Overview

<<<PAGE 2>>>
### Sub-task Status

<<<PAGE 3>>>
### 3. New Items

#### Architectural Discrepancy

<<<PAGE 4>>>
# 4. Action Items for Next Meeting

## Notes
"""


def test_restructure_normalizes_inconsistent_levels():
    out = restructure_markdown(RAW_INCONSISTENT_MD)
    lines = out.splitlines()
    # exactly one '# ' title, and it's the document title (not a later running head / numbered section)
    h1 = [ln for ln in lines if ln.startswith("# ")]
    assert h1 == ["# Weekly Meeting Update"], h1
    # every numbered top-level section lands at '## ' regardless of the level the model emitted
    for sec in ("## 1. Action Items", "## 2. Milestone Overview", "## 3. New Items",
                "## 4. Action Items for Next Meeting"):
        assert sec in out, sec
    # non-canonical / decimal-ish sub-headings stay '### '
    assert "### Sub-task Status" in out
    assert "### Architectural Discrepancy" in out      # was '####', demoted to '###'
    assert "#### " not in out                            # no level-4+ headings remain


def test_restructure_demotes_stray_h1_running_head():
    raw = "# Real Title\n\nbody\n\n<<<PAGE 2>>>\n# Smith et al.\n\nmore body\n"
    out = restructure_markdown(raw)
    assert out.count("\n# ") + out.startswith("# ") == 1  # single H1
    assert "### Smith et al." in out                      # stray '#' running head demoted


def test_restructure_leaves_tables_latex_and_code_fences_untouched():
    raw = (
        "# Title\n\n## Methods\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\n"
        "Energy $E = mc^2$.\n\n```python\n# this is code, not a heading\nx = 1\n```\n"
    )
    out = restructure_markdown(raw)
    assert "| A | B |" in out and "| 1 | 2 |" in out      # table preserved
    assert "$E = mc^2$" in out                             # LaTeX preserved
    assert "# this is code, not a heading" in out          # '#' inside a code fence NOT re-levelled


def test_flatten_headings_collapses_all_levels_to_two():
    out = flatten_headings(RAW_NATIVE_MD)
    assert "# A Faithful OCR" not in out.splitlines()[0] or out.splitlines()[0].startswith("## ")
    assert "## A Faithful OCR of a Manuscript" in out  # '#' -> '##'
    assert "## Methods" in out
    assert "## Cell Culture" in out                    # '###' -> '##'
    assert "### " not in out and "\n# " not in ("\n" + out)


def test_format_document_dispatch():
    assert format_document(RAW, "headers") == to_headers(RAW)
    assert format_document(RAW, "md") == to_markdown(RAW)
    # native_markdown routing
    assert format_document(RAW_NATIVE_MD, "md", native_markdown=True) == restructure_markdown(RAW_NATIVE_MD)
    assert format_document(RAW_NATIVE_MD, "headers", native_markdown=True) == flatten_headings(RAW_NATIVE_MD)
    try:
        format_document(RAW, "bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown format")


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
    print("ok — all section tests pass")
