"""Command-line interface: ``scientificpub2md PDF [PDF ...] [options]``."""
from __future__ import annotations

import argparse
import os
import sys
import time

from . import __version__
from .extract import make_backend
from .sections import format_document

_EXT = {"md": ".md", "headers": ".txt", "clean": ".clean.txt"}


def _gather_inputs(paths):
    """Expand directories to the PDFs inside them; keep file paths as given."""
    pdfs = []
    for p in paths:
        if os.path.isdir(p):
            pdfs += sorted(
                os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(".pdf")
            )
        else:
            pdfs.append(p)
    return pdfs


def _out_path(pdf, out, fmt, many):
    stem = os.path.splitext(os.path.basename(pdf))[0]
    ext = _EXT[fmt]
    if out is None:
        return os.path.join(os.path.dirname(pdf) or ".", stem + ext)
    if many or os.path.isdir(out) or out.endswith(os.sep):
        os.makedirs(out, exist_ok=True)
        return os.path.join(out, stem + ext)
    return out  # single explicit file


def build_parser():
    ap = argparse.ArgumentParser(
        prog="scientificpub2md",
        description="Lightweight VLM OCR: scientific manuscript PDFs -> clean Markdown or '## ' headers.",
    )
    ap.add_argument("inputs", nargs="+", help="PDF file(s) or a directory of PDFs")
    ap.add_argument("-o", "--out", default=None,
                    help="output file (single PDF) or directory (multiple); default: alongside each PDF")
    ap.add_argument("-f", "--format", choices=["md", "headers", "clean"], default="md",
                    help="md = structured markdown (.md); headers = flat '## ' headers (.txt); "
                         "clean = junk-stripped, '## '-denoted, section_map-ready text with inferred "
                         "headers (.clean.txt). Default: md")
    ap.add_argument("-e", "--engine", choices=["lightonocr", "qwen3vl"], default="lightonocr",
                    help="lightonocr = LightOnOCR-2-1B (1B, fast, Mac-friendly, native tables/LaTeX; no extra deps); "
                         "qwen3vl = Qwen3-VL-8B + scientific prompt (drops back matter; needs [qwen] extra for the "
                         "transformers backend). Default: lightonocr")
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto",
                    help="compute device for the transformers backend (auto picks CUDA, then Apple MPS, then CPU)")
    ap.add_argument("--backend", choices=["transformers", "vllm"], default="transformers",
                    help="transformers = in-process (CPU/GPU/MPS); vllm = local vLLM HTTP server (GPU). Default: transformers")
    ap.add_argument("--model", default=None, help="override the VLM model id (defaults per engine)")
    ap.add_argument("--dpi", type=int, default=None, help="page render DPI (default: 170 qwen3vl / 200 lightonocr)")
    ap.add_argument("--max-pages", type=int, default=None, help="limit pages per PDF (debugging)")
    ap.add_argument("--workers", type=int, default=8, help="concurrent pages for the vllm backend (default: 8)")
    ap.add_argument("--keep-backmatter", action="store_true",
                    help="keep References/Acknowledgements/Funding etc. (default: drop them)")
    ap.add_argument("--vllm-url", default=None, help="vLLM server URL (default: $SCIPUB2MD_VLLM_URL or http://localhost:8000)")
    ap.add_argument("--vllm-model", default="qwen3-vl-8b", help="served model name on the vLLM server")
    ap.add_argument("-q", "--quiet", action="store_true", help="suppress per-page progress")
    ap.add_argument("-V", "--version", action="version", version=f"scientificpub2md {__version__}")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    pdfs = _gather_inputs(args.inputs)
    if not pdfs:
        print("No PDFs found.", file=sys.stderr)
        return 2
    many = len(pdfs) > 1
    verbose = not args.quiet

    backend = make_backend(
        args.backend, engine=args.engine, device=args.device, model=args.model,
        keep_backmatter=args.keep_backmatter, vllm_url=args.vllm_url, vllm_model=args.vllm_model,
    )
    native_md = getattr(backend, "native_markdown", False)
    print(f"scientificpub2md {__version__}: {len(pdfs)} PDF(s), engine={args.engine}, "
          f"backend={args.backend}, format={args.format}, device={getattr(backend, 'device', '?')}", flush=True)

    # Import here so the lighter modules import fast.
    from .extract import extract_pdf_concurrent

    rc = 0
    t0 = time.time()
    for i, pdf in enumerate(pdfs, 1):
        if not os.path.exists(pdf):
            print(f"  [{i}/{len(pdfs)}] MISSING: {pdf}", file=sys.stderr)
            rc = 1
            continue
        outp = _out_path(pdf, args.out, args.format, many)
        if verbose:
            print(f"  [{i}/{len(pdfs)}] {os.path.basename(pdf)} -> {outp}", flush=True)
        try:
            raw = extract_pdf_concurrent(
                pdf, backend, dpi=args.dpi, max_pages=args.max_pages,
                workers=args.workers, verbose=verbose,
            )
            doc = format_document(raw, fmt=args.format, native_markdown=native_md)
            with open(outp, "w") as fh:
                fh.write(doc)
            if verbose:
                print(f"      wrote {len(doc)} chars", flush=True)
        except Exception as e:
            print(f"      FAILED: {e}", file=sys.stderr)
            rc = 1
    if verbose:
        print(f"Done in {time.time() - t0:.0f}s", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
