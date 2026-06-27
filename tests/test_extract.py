"""Tests for the extraction layer — model-free, so they run in CI without downloading any weights.

A ``FakeBackend`` stands in for a real VLM: it returns canned per-page text instead of transcribing
the rendered image. That exercises the real render → transcribe → assemble path (page markers, page
ordering, no-truncation, ``max_pages``, and the ``convert_pdf`` format routing) against a genuine
PDF built on the fly with PyMuPDF. It also shows the "golden" fuzzy-match pattern an OCR regression
suite would use to catch quality drift across model/transformers bumps.
"""
import os
import sys
import tempfile
from difflib import SequenceMatcher

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scientificpub2md import convert_pdf, extract_pdf  # noqa: E402
from scientificpub2md.extract import extract_pdf_concurrent  # noqa: E402


class FakeBackend:
    """A backend that returns canned text per page (call order == page order for the in-process path)."""

    default_dpi = 72

    def __init__(self, pages, *, native_markdown=False):
        self._pages = pages
        self.native_markdown = native_markdown
        self.device = "cpu (fake)"
        self.calls = 0

    def transcribe(self, _pil_img, max_new_tokens=4096):
        text = self._pages[self.calls]
        self.calls += 1
        return text


class FakeBatchBackend(FakeBackend):
    """Adds a transcribe_batch so the batched code path can be exercised model-free."""

    def __init__(self, pages, **kw):
        super().__init__(pages, **kw)
        self.batch_calls = 0
        self.max_batch = 0

    def transcribe_batch(self, pil_imgs, max_new_tokens=4096):
        self.batch_calls += 1
        self.max_batch = max(self.max_batch, len(pil_imgs))
        return [self.transcribe(im) for im in pil_imgs]


def _make_pdf(n_pages):
    import fitz

    path = os.path.join(tempfile.mkdtemp(), "synthetic.pdf")
    doc = fitz.open()
    for i in range(n_pages):
        doc.new_page().insert_text((72, 72), f"page {i + 1} body")
    doc.save(path)
    doc.close()
    return path


def test_page_markers_and_order_and_no_truncation():
    pdf = _make_pdf(3)
    pages = ["## Intro\nalpha", "## Methods\nbeta", "## Results\ngamma"]
    raw = extract_pdf(pdf, FakeBackend(pages), verbose=False)
    # every page is present, in order, each behind its own marker — nothing truncated
    for n in (1, 2, 3):
        assert f"<<<PAGE {n}>>>" in raw
    assert raw.index("alpha") < raw.index("beta") < raw.index("gamma")
    assert "## Results" in raw and "gamma" in raw


def test_max_pages_limits_transcription():
    pdf = _make_pdf(5)
    be = FakeBackend(["p1", "p2", "p3", "p4", "p5"])
    raw = extract_pdf(pdf, be, max_pages=2, verbose=False)
    assert be.calls == 2                      # only two pages transcribed
    assert "<<<PAGE 2>>>" in raw and "<<<PAGE 3>>>" not in raw


def test_concurrent_falls_back_to_sequential_for_inprocess():
    pdf = _make_pdf(2)
    raw = extract_pdf_concurrent(pdf, FakeBackend(["one", "two"]), verbose=False)
    assert "<<<PAGE 1>>>" in raw and "<<<PAGE 2>>>" in raw
    assert "one" in raw and "two" in raw


def test_convert_pdf_format_routing():
    pdf = _make_pdf(2)
    pages = ["# Big Title\n\n## Methods\nbody one", "### 2. Findings\nbody two"]
    # native-markdown 'md' re-levels into a consistent hierarchy
    md = convert_pdf(pdf, FakeBackend(pages, native_markdown=True), fmt="md", verbose=False)
    assert md.count("\n# ") + md.startswith("# ") == 1   # single H1 title
    assert "## Methods" in md and "## 2. Findings" in md  # both top-level sections at '##'
    # 'headers' flattens everything to '##'
    headers = convert_pdf(pdf, FakeBackend(pages, native_markdown=True), fmt="headers", verbose=False)
    assert "### " not in headers and "\n# " not in ("\n" + headers)


def test_batched_extraction_matches_sequential_and_preserves_order():
    pdf = _make_pdf(5)
    pages = ["## A\nuno", "## B\ndos", "## C\ntres", "## D\ncuatro", "## E\ncinco"]
    seq = extract_pdf(pdf, FakeBatchBackend(pages), verbose=False)          # batch_size=1 default
    be = FakeBatchBackend(pages)
    batched = extract_pdf(pdf, be, verbose=False, batch_size=2)
    assert batched == seq                       # identical output regardless of batching
    assert be.batch_calls == 3                  # 2 + 2 + 1
    assert be.max_batch == 2                     # never exceeds batch_size
    assert "uno" in batched and "cinco" in batched
    assert batched.index("uno") < batched.index("cinco")  # order preserved


def test_batch_size_falls_back_when_backend_has_no_transcribe_batch():
    pdf = _make_pdf(2)
    # plain FakeBackend has no transcribe_batch -> must still work via per-page path
    out = extract_pdf(pdf, FakeBackend(["x", "y"]), verbose=False, batch_size=4)
    assert "<<<PAGE 1>>>" in out and "<<<PAGE 2>>>" in out and "x" in out and "y" in out


def test_golden_fuzzy_match_pattern():
    """Demonstrates the OCR-regression pattern: extracted body must stay close to expected text."""
    pdf = _make_pdf(2)
    pages = ["Heterocytes are specialized cells for nitrogen fixation.",
             "We found a conserved biosynthetic gene cluster."]
    raw = extract_pdf(pdf, FakeBackend(pages), verbose=False)
    body = raw.replace("<<<PAGE 1>>>", "").replace("<<<PAGE 2>>>", "")
    expected = "\n".join(pages)
    ratio = SequenceMatcher(None, expected.split(), body.split()).ratio()
    assert ratio > 0.9, f"extracted text drifted from golden (ratio={ratio:.2f})"


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
    print("ok — all extract tests pass")
