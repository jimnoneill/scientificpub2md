"""Tests for the deterministic clean/normalize layer (no model needed)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scientificpub2md.clean import DEFAULT_SCRUB, _classify_backmatter, clean_document  # noqa: E402

# Faithful-OCR style output: native markdown levels, banners, running heads, page numbers,
# a run-in header, an unmarked header, and a full back-matter tail.
RAW = """\

<<<PAGE 1>>>
# PNAS
## RESEARCH ARTICLE | MICROBIOLOGY
### OPEN ACCESS

# Heterocyte glycolipid biosynthesis in cyanobacteria

Edited by Roger Summons; received July 13, 2024; accepted December 3, 2024

## Abstract
Heterocytes are specialized cells.

Smith et al. | PNAS | 2024
1

<<<PAGE 2>>>
Smith et al. | PNAS | 2024
2

Introduction: Nitrogen fixation is central to cyanobacterial ecology and has been studied for decades.

Materials and Methods

### Statistical Analysis

We used t-tests. doi:10.1234/abcd

<<<PAGE 3>>>
Smith et al. | PNAS | 2024
3

## Results
We found things. See https://example.org/data

## Discussion
It means something.

## Data Availability
Sequences are in GenBank.

## Funding
Funded by NSF grant 123.

## Acknowledgements
We thank the lab.

## References
1. Author A. Title. Journal. 2020.
2. Author B. Title. Journal. 2021.
"""


def _run():
    return clean_document(RAW)


def test_drops_running_heads_and_page_numbers():
    out = _run()
    assert "Smith et al. | PNAS | 2024" not in out   # running head (repeats on 3 pages)
    assert "\n1\n" not in ("\n" + out + "\n")          # bare page numbers gone


def test_drops_front_matter_banners_and_meta():
    out = _run()
    assert "RESEARCH ARTICLE" not in out
    assert "OPEN ACCESS" not in out
    assert "PNAS" not in out                            # single all-caps journal masthead dropped
    assert "Edited by Roger Summons" not in out        # editorial metadata line


def test_normalizes_headings_to_flat_h2():
    out = _run()
    # the title was '# ' and a banner was '### ' — everything kept becomes '## '
    assert "## Heterocyte glycolipid biosynthesis in cyanobacteria" in out
    assert "\n# " not in ("\n" + out)                  # no level-1 headings remain
    assert "### " not in out                            # no level-3 headings remain


def test_infers_missing_headers():
    out = _run()
    assert "## Introduction" in out                     # run-in header split out
    assert "Nitrogen fixation is central" in out        # its body preserved
    assert "## Materials and Methods" in out            # bare vocab line promoted to a header
    assert "## Statistical Analysis" in out             # '### ' sub-heading normalized to '## '


def test_scrubs_full_backmatter_set_keeps_body_and_abstract():
    out = _run()
    # kept
    assert "## Abstract" in out and "Heterocytes are specialized cells." in out
    assert "## Results" in out and "## Discussion" in out
    # scrubbed (full Qwen set)
    for gone in ("## References", "## Acknowledgements", "## Funding", "## Data Availability",
                 "Sequences are in GenBank", "Funded by NSF", "We thank the lab", "Author A. Title"):
        assert gone not in out, f"should have scrubbed: {gone!r}"


def test_default_scrub_is_full_qwen_set():
    assert set(DEFAULT_SCRUB) == {
        "references", "acknowledgements", "funding", "author contributions",
        "competing interests", "data availability", "supplementary",
    }


def test_classifier_categories():
    assert _classify_backmatter("References") == "references"
    assert _classify_backmatter("Acknowledgments") == "acknowledgements"   # US spelling
    assert _classify_backmatter("Author Contributions") == "author contributions"
    assert _classify_backmatter("Competing Interests") == "competing interests"
    assert _classify_backmatter("Declaration of Competing Interest") == "competing interests"
    assert _classify_backmatter("Data and Code Availability") == "data availability"
    assert _classify_backmatter("Supplementary Information") == "supplementary"
    assert _classify_backmatter("Methods") == "other"
    assert _classify_backmatter("Results") == "other"


def test_configurable_scrub_keeps_unlisted():
    out = clean_document(RAW, scrub_sections=("references",))
    assert "## References" not in out
    assert "## Funding" in out                          # not scrubbed when not requested
    assert "## Acknowledgements" in out


def test_verbatim_body_preserved():
    out = _run()
    assert "We found things." in out                    # body words untouched
    assert "It means something." in out


# A faithful OCR (LightOnOCR) emits tables as HTML. The same structural tags recur on every
# table-bearing page, so the running-head detector must NOT treat them as furniture and shred them.
RAW_TABLES = """\

<<<PAGE 1>>>
## Results
<table>
  <thead><tr><th>Gene</th><th>Fold change</th></tr></thead>
  <tbody><tr><td>hglB</td><td>2.1</td></tr></tbody>
</table>

<<<PAGE 2>>>
## Discussion
<table>
  <thead><tr><th>Strain</th><th>Phenotype</th></tr></thead>
  <tbody><tr><td>WT</td><td>het+</td></tr></tbody>
</table>

<<<PAGE 3>>>
## Conclusions
<table>
  <thead><tr><th>Model</th><th>Score</th></tr></thead>
  <tbody><tr><td>ours</td><td>0.91</td></tr></tbody>
</table>
"""


def test_html_tables_survive_furniture_detection():
    out = clean_document(RAW_TABLES)
    # structural tags repeat on all 3 pages but must be kept (not dropped as running heads)
    assert out.count("<table>") == 3, "table opening tags were dropped"
    assert out.count("</table>") == 3, "table closing tags were dropped"
    assert "<tr>" in out and "<thead>" in out, "table structure shredded"
    # and no cell content is orphaned outside a table
    assert "hglB" in out and "0.91" in out


# A non-paper document (no Abstract/Introduction/Methods/… heading) so seen_section never flips.
# Front-matter-only filters must disarm after the window, or they'd eat body lines all the way down.
RAW_NONPAPER = """\

<<<PAGE 1>>>
## Weekly Report
Correspondence about this report goes to the front desk.
body line a
body line b
body line c
Published online dashboards are now available to all staff this quarter.
"""


def test_front_matter_window_disarms_for_nonpaper_docs():
    # tiny window: the early "Correspondence …" line is still in front matter (dropped),
    # but the later "Published online …" body line is past the window and kept.
    out = clean_document(RAW_NONPAPER, front_matter_window=3)
    assert "Correspondence about this report" not in out   # within window -> dropped as meta
    assert "Published online dashboards are now available" in out  # past window -> kept
    assert "## Weekly Report" in out


def test_front_matter_filters_still_apply_at_top():
    # with a generous window the meta line at the very top is dropped, as before
    out = clean_document(RAW_NONPAPER)
    assert "Correspondence about this report" not in out
    assert "body line a" in out


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ok — all clean tests pass")
