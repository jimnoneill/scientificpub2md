<p align="center">
  <img src="https://raw.githubusercontent.com/jimnoneill/scientificpub2md/main/docs/assets/logo.png" alt="scientificpub2md" width="120">
</p>

<h1 align="center">scientificpub2md</h1>

<p align="center">
  <strong>Lightweight VLM OCR that turns scientific manuscript PDFs into clean Markdown — on CPU, GPU, or Apple Silicon.</strong>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#two-output-formats">Formats</a> ·
  <a href="#two-engines">Engines</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#when-something-goes-wrong">Troubleshooting</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/scientificpub2md/"><img src="https://img.shields.io/pypi/v/scientificpub2md?style=flat-square&logo=pypi&logoColor=white&color=4f46e5" alt="PyPI version"></a>
  <a href="https://pepy.tech/project/scientificpub2md"><img src="https://img.shields.io/pepy/dt/scientificpub2md?style=flat-square&color=06b6d4&label=downloads" alt="Total downloads"></a>
  <img src="https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Engine-Qwen3--VL%20%7C%20LightOnOCR-4f46e5?style=flat-square" alt="Engines">
  <img src="https://img.shields.io/badge/Runs%20on-CPU%20%7C%20GPU%20%7C%20Apple%20MPS-06b6d4?style=flat-square" alt="Runs on CPU, GPU, MPS">
  <img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" alt="License: MIT">
  <a href="https://paypal.me/jimnoneill"><img src="https://img.shields.io/badge/Donate-PayPal-00457C?style=flat-square&logo=paypal" alt="Donate via PayPal"></a>
</p>

---

## Why

Most PDF parsers choke on real scientific papers: they mangle multi-column
reading order, silently drop the detailed methods, or truncate long
manuscripts. `scientificpub2md` takes a different route — it **renders each
page to an image and transcribes it one page at a time with a vision-language
model**, then stitches the pages back together. Nothing is truncated however
long the paper, the reading order is the one a human sees, and every section
heading comes out as a clean Markdown header.

```
  PDF ──render pages (PyMuPDF)──▶ page images ──VLM transcribe──▶ clean text ──▶ .md  or  ## headers
```

- **Verbatim, not summarized** — the model transcribes the page, it doesn't paraphrase.
- **No truncation** — pages are processed individually, so a 50-page paper is fully captured.
- **Headings preserved** — every section and sub-section becomes a Markdown header.
- **Deterministic** — greedy decoding means re-extracting a PDF is byte-reproducible.
- **Runs anywhere** — in-process on CPU, NVIDIA GPU, or Apple Silicon (MPS); or via a fast batched vLLM server.

## Quick start

```bash
pip install git+https://github.com/jimnoneill/scientificpub2md
```

```bash
# Single PDF → Markdown (default engine: LightOnOCR-2-1B; auto-selects GPU → Apple MPS → CPU)
scientificpub2md paper.pdf

# A whole directory of PDFs into an output folder
scientificpub2md ./pdfs/ -o ./markdown/

# Simple flat "## " headers instead of structured markdown
scientificpub2md paper.pdf --format headers

# Use the Qwen3-VL-8B scientific engine (drops references, preserves methods)
pip install "scientificpub2md[qwen] @ git+https://github.com/jimnoneill/scientificpub2md"
scientificpub2md paper.pdf --engine qwen3vl
```

Output lands next to each PDF (`paper.pdf → paper.md`); use `-o` for an explicit file or directory.

From Python:

```python
from scientificpub2md import pdf_to_markdown

md = pdf_to_markdown("paper.pdf")                          # default (lightonocr), auto device
md = pdf_to_markdown("paper.pdf", engine="qwen3vl")        # Qwen3-VL-8B + scientific prompt
```

## Two output formats

| Format | Flag | Ext | What you get |
|---|---|---|---|
| **Markdown** (default) | `--format md` | `.md` | Structured: title → `#`, canonical sections (Abstract, Methods, Results, …) → `##`, sub-headings → `###`. |
| **Simple headers** | `--format headers` | `.txt` | The clean full text with every heading flat at `##`. Minimal. |
| **Clean** | `--format clean` | `.clean.txt` | Junk-stripped, flat-`##`, **verbatim** text with missing headers inferred — ready for a downstream section/feature pipeline. |

The heading levelling is **deterministic** (a section-name vocabulary, no extra LLM call), so output is reproducible.

### The `clean` format — deterministic, pipeline-ready

`clean` is built for feeding a downstream parser (it's the layer that lets a faithful OCR like
LightOnOCR stand in for an editorial, prompt-steered VLM). It applies, with **no model** (every
step is a rule, so it's byte-reproducible and never rewrites words — lines are only dropped and
headings only marked):

- drops running heads/footers (lines repeating across pages), page/line numbers, horizontal rules, DOI/copyright-only lines, and front-matter banners (`OPEN ACCESS`, `RESEARCH ARTICLE`, journal mastheads like `PNAS`)
- scrubs back-matter sections — references, acknowledgements, funding, author contributions, competing interests, data availability, supplementary (configurable via `clean_document(..., scrub_sections=...)`)
- normalizes every heading to a flat `## `
- **infers missing headers** — promotes unmarked section-vocabulary lines and splits run-in headers (`Introduction: …`) onto their own `## ` line, the way a prompted VLM would

```python
from scientificpub2md import make_backend, extract_pdf, clean_document
raw = extract_pdf("paper.pdf", make_backend("transformers", engine="lightonocr"))
text = clean_document(raw)                 # verbatim, junk-stripped, '## '-denoted
```

## Two engines

Pick with `--engine` (or `engine=` in Python). They're complementary — a tiny specialized OCR model, and a steerable general VLM tuned for papers.

| | `--engine lightonocr` *(default)* | `--engine qwen3vl` |
|---|---|---|
| Model | LightOnOCR-2-1B (1B, purpose-built OCR) | Qwen3-VL-8B (8B, general VLM + scientific prompt) |
| Install | base (`pip install scientificpub2md`) | needs `[qwen]` extra (torchvision) for transformers |
| Memory | ~2–3 GB — **great on a laptop** | ~16 GB (bf16) |
| Speed | much faster (~5 pages/s on an H100) | slower |
| Editorial smarts | faithful full-page transcription (keeps everything) | **drops references / acknowledgements / page furniture**, preserves methods |
| Tables & equations | **native Markdown tables + LaTeX** | as plain text |
| Languages | 11 languages | English-tuned prompt |
| Best for | fast/cheap transcription, tables/math, Macs & small GPUs | scientific-corpus building, back-matter dropping, reproducibility |

> **macOS / Apple Silicon:** both engines run on MPS, but the 8B Qwen model is heavy for most Macs — the default `lightonocr` is small and fast on a MacBook.

<details>
<summary>Example output (<code>--format md</code>)</summary>

```markdown
# Emergence and evolution of heterocyte glycolipid biosynthesis in cyanobacteria

## Abstract
Heterocytes, specialized cells for nitrogen fixation in cyanobacteria, are …

## Results and Discussion

### Genomic Prediction of HG Biosynthesis
To investigate the evolution of HG biosynthesis within cyanobacteria, we searched …
```
</details>

## Install options

Requires **Python 3.10+**.

```bash
# Simplest — base install (default lightonocr engine), CPU/MPS wheels:
pip install git+https://github.com/jimnoneill/scientificpub2md

# NVIDIA GPU — install a CUDA build of torch first, then the package:
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match your CUDA
pip install git+https://github.com/jimnoneill/scientificpub2md

# Add the Qwen3-VL engine's transformers backend (pulls torchvision — match your torch build):
pip install "scientificpub2md[qwen] @ git+https://github.com/jimnoneill/scientificpub2md"

# Fast batched serving (advanced):
pip install "scientificpub2md[vllm] @ git+https://github.com/jimnoneill/scientificpub2md"
```

> The `qwen3vl` engine's in-process backend needs `torchvision` — install `torch` and
> `torchvision` from the **same** index (e.g. both from the CUDA index) or they won't load.
> `lightonocr` and the vLLM backend need no torchvision.

| Package | Why |
|---|---|
| `PyMuPDF` | render PDF pages → images |
| `Pillow` | image handling |
| `transformers` (≥4.57) | runs the models (LightOnOCR needs a recent build; older falls back to remote code) |
| `torch` (≥2.4) | model runtime — **install the CUDA build for NVIDIA GPUs** |
| `accelerate` | device placement |
| `requests` | client for the optional vLLM backend |
| `torchvision` *(extra `[qwen]`)* | required by the Qwen3-VL processor |
| `vllm` *(extra `[vllm]`)* | fast batched GPU serving |

### Fast batched throughput (optional)

For many papers, serve a model once and let vLLM batch the in-flight page requests:

```bash
./serve_vllm.sh                                              # Qwen3-VL-8B on :8000
SCIPUB2MD_VLM_ID=lightonai/LightOnOCR-2-1B ./serve_vllm.sh   # or LightOnOCR

scientificpub2md ./pdfs/ -o ./md/ --backend vllm --workers 16            # qwen3vl
scientificpub2md ./pdfs/ -o ./md/ --backend vllm --engine lightonocr     # lightonocr
```

The server can be remote — set `SCIPUB2MD_VLLM_URL=http://host:8000` (or pass `--vllm-url`).

## How it works

1. **Render** — each PDF page → a PNG with PyMuPDF (170 DPI qwen3vl / 200 DPI lightonocr).
2. **Transcribe** — each page image → text with the chosen model, greedy-decoded (deterministic).
3. **Assemble** — pages concatenated in order; nothing truncated.
4. **Format** — restructured into `# / ## / ###` Markdown, or flattened to simple `##` headers.

## When something goes wrong

| Symptom | Likely cause / fix |
|---|---|
| `device='cuda' requested but no CUDA GPU` | No GPU visible to torch — use `--device cpu`/`mps`, or install a CUDA torch build |
| Out-of-memory loading qwen3vl | The 8B model needs ~16 GB — use `--engine lightonocr` or a smaller `--device` |
| `qwen3vl … needs torchvision` | `pip install 'scientificpub2md[qwen]'` (match torch/torchvision builds), or use `--engine lightonocr` |
| `operator torchvision::nms does not exist` | torch/torchvision mismatch — reinstall both from the same index |
| Small text / subscripts garbled | Raise `--dpi` (e.g. `--dpi 220`) |
| Connection refused on `--backend vllm` | No server — run `./serve_vllm.sh`, or set `--vllm-url` |
| Very slow on CPU | Expected for the 8B model — use `--engine lightonocr` or a GPU |

## Notes & limitations

- **OCR quality tracks the model.** Dense sub/superscripts, exotic glyphs, and complex tables can have errors. Raise `--dpi` if small text is missed.
- **`qwen3vl`** drops back matter and page furniture by default (`--keep-backmatter` to retain). **`lightonocr`** is a faithful transcriber and keeps everything (not prompt-steerable).
- **Tables/figures** under qwen3vl come out as plain text; lightonocr reconstructs Markdown tables and LaTeX.
- Determinism assumes a fixed model + prompt + DPI.

## Support

If this saved you from wrangling GROBID or hand-cleaning PDFs, contributions
toward continued maintenance are welcome.

<p>
  <a href="https://paypal.me/jimnoneill"><img src="https://img.shields.io/badge/Donate-PayPal-00457C?style=for-the-badge&logo=paypal" alt="Donate via PayPal"></a>
</p>

## License

MIT — see [LICENSE](LICENSE).

*Born out of the [PubVerse](https://github.com/jimnoneill) extraction pipeline, where this page-by-page VLM method replaced GROBID for building clean full-text scientific corpora.*
