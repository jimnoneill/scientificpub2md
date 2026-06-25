"""scientificpub2md — lightweight VLM OCR that turns scientific manuscript PDFs into clean
Markdown (.md) or simple ``## `` headers, on CPU or GPU.

Quick start::

    from scientificpub2md import pdf_to_markdown
    md = pdf_to_markdown("paper.pdf")            # transformers backend, auto CPU/GPU

Or build a backend once and reuse it across many PDFs::

    from scientificpub2md import make_backend, convert_pdf
    backend = make_backend("transformers", device="cuda")
    md = convert_pdf("paper.pdf", backend, fmt="md")
"""
from .extract import (
    TransformersBackend,
    VLLMBackend,
    extract_pdf,
    extract_pdf_concurrent,
    make_backend,
    render_pages,
)
from .sections import format_document, to_headers, to_markdown

__version__ = "0.1.0"

__all__ = [
    "TransformersBackend",
    "VLLMBackend",
    "make_backend",
    "render_pages",
    "extract_pdf",
    "extract_pdf_concurrent",
    "to_markdown",
    "to_headers",
    "format_document",
    "convert_pdf",
    "pdf_to_markdown",
]


def convert_pdf(pdf_path, backend, fmt="md", *, dpi=170, max_pages=None, workers=8, verbose=True):
    """Extract a PDF with ``backend`` and return it formatted as 'md' or 'headers'."""
    raw = extract_pdf_concurrent(
        pdf_path, backend, dpi=dpi, max_pages=max_pages, workers=workers, verbose=verbose
    )
    return format_document(raw, fmt=fmt)


def pdf_to_markdown(pdf_path, *, device="auto", model=None, fmt="md", dpi=170, max_pages=None,
                    keep_backmatter=False, verbose=True):
    """One-call convenience: extract ``pdf_path`` with the in-process transformers backend.

    Returns the document as Markdown ('md', default) or simple ``## `` headers ('headers').
    """
    backend = make_backend(
        "transformers", device=device, model=model, keep_backmatter=keep_backmatter
    )
    return convert_pdf(pdf_path, backend, fmt=fmt, dpi=dpi, max_pages=max_pages, verbose=verbose)
