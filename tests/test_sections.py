"""Tests for the format layer (no model needed — pure text transforms)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scientificpub2md.sections import format_document, to_headers, to_markdown  # noqa: E402

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


def test_format_document_dispatch():
    assert format_document(RAW, "headers") == to_headers(RAW)
    assert format_document(RAW, "md") == to_markdown(RAW)
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
    test_format_document_dispatch()
    print("ok — all section tests pass")
