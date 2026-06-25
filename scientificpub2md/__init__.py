"""scientificpub2md — lightweight VLM OCR that turns scientific manuscript PDFs into clean
Markdown (.md) or simple ``## `` headers, on CPU or GPU.

Two engines:
  * ``lightonocr`` (default) — LightOnOCR-2-1B, a purpose-built 1B OCR model (native
    Markdown/tables/LaTeX); runs out of the box on CPU/GPU/Apple Silicon.
  * ``qwen3vl`` — Qwen3-VL-8B steered by a scientific-manuscript prompt (drops back matter,
    preserves methods); the in-process backend needs the ``[qwen]`` extra (torchvision).

Quick start::

    from scientificpub2md import pdf_to_markdown
    md = pdf_to_markdown("paper.pdf")                          # lightonocr, auto CPU/GPU/MPS
    md = pdf_to_markdown("paper.pdf", engine="qwen3vl")        # Qwen3-VL-8B + scientific prompt

Or build a backend once and reuse it across many PDFs::

    from scientificpub2md import make_backend, convert_pdf
    backend = make_backend("transformers", engine="lightonocr", device="cuda")
    md = convert_pdf("paper.pdf", backend, fmt="md")
"""
from .extract import (
    LightOnOCRBackend,
    TransformersBackend,
    VLLMBackend,
    extract_pdf,
    extract_pdf_concurrent,
    make_backend,
    render_pages,
)
from .sections import (
    flatten_headings,
    format_document,
    passthrough_markdown,
    to_headers,
    to_markdown,
)

__version__ = "0.2.0"

__all__ = [
    "TransformersBackend",
    "LightOnOCRBackend",
    "VLLMBackend",
    "make_backend",
    "render_pages",
    "extract_pdf",
    "extract_pdf_concurrent",
    "to_markdown",
    "to_headers",
    "flatten_headings",
    "passthrough_markdown",
    "format_document",
    "convert_pdf",
    "pdf_to_markdown",
]


def convert_pdf(pdf_path, backend, fmt="md", *, dpi=None, max_pages=None, workers=8, verbose=True):
    """Extract a PDF with ``backend`` and return it formatted as 'md' or 'headers'.

    The right format treatment is chosen from the backend's output convention
    (``backend.native_markdown``): native-Markdown engines pass through / flatten, while the
    flat-``## `` engine is restructured.
    """
    raw = extract_pdf_concurrent(
        pdf_path, backend, dpi=dpi, max_pages=max_pages, workers=workers, verbose=verbose
    )
    return format_document(raw, fmt=fmt, native_markdown=getattr(backend, "native_markdown", False))


def pdf_to_markdown(pdf_path, *, engine="lightonocr", device="auto", model=None, fmt="md", dpi=None,
                    max_pages=None, keep_backmatter=False, verbose=True):
    """One-call convenience: extract ``pdf_path`` with the in-process transformers backend.

    engine='lightonocr' (default) or 'qwen3vl'. Returns the document as Markdown ('md', default)
    or simple ``## `` headers ('headers').
    """
    backend = make_backend(
        "transformers", engine=engine, device=device, model=model, keep_backmatter=keep_backmatter
    )
    return convert_pdf(pdf_path, backend, fmt=fmt, dpi=dpi, max_pages=max_pages, verbose=verbose)
