"""Tests for the format layer (no model needed — pure text transforms)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scientificpub2md.sections import (  # noqa: E402
    flatten_headings,
    format_document,
    passthrough_markdown,
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
    assert format_document(RAW_NATIVE_MD, "md", native_markdown=True) == passthrough_markdown(RAW_NATIVE_MD)
    assert format_document(RAW_NATIVE_MD, "headers", native_markdown=True) == flatten_headings(RAW_NATIVE_MD)
    try:
        format_document(RAW, "bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown format")


if __name__ == "__main__":
    test_headers_strips_page_markers_and_keeps_flat()
    test_markdown_promotes_title_and_demotes_subheadings()
    test_markdown_promotes_title_marked_as_heading()
    test_native_markdown_passthrough_preserves_levels_and_tables()
    test_flatten_headings_collapses_all_levels_to_two()
    test_format_document_dispatch()
    print("ok — all section tests pass")
