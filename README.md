# scientificpub2md

**Lightweight VLM OCR that turns scientific manuscript PDFs into clean Markdown (`.md`) or simple `##` headers — on CPU or GPU.**

Most PDF parsers choke on multi-column scientific layouts, drop detailed methods, mangle reading order, or truncate long papers. `scientificpub2md` takes a different approach: it **renders each page to an image and transcribes it one page at a time with a vision-language model** (Qwen3-VL-8B by default). Nothing is truncated however long the paper, the reading order is whatever a human sees, and every section heading comes out as a clean `##` markdown header.

```
PDF ──render pages (PyMuPDF)──▶ page images ──VLM transcribe──▶ clean text with ## headers ──▶ .md  or  ## headers
```

- **Verbatim, not summarized.** The model transcribes the page; it does not paraphrase.
- **No truncation.** Pages are processed individually and concatenated, so a 50-page paper is fully captured.
- **Headings preserved.** Every section and sub-section is marked `## …` (multi-column reading order handled).
- **Noise dropped.** Running heads, page numbers, journal banners, DOIs and (by default) the back matter — references, acknowledgements, funding — are omitted.
- **Deterministic.** Greedy decoding (temperature 0) means re-extracting a PDF is byte-reproducible.
- **CPU or GPU.** Runs in-process via 🤗 transformers on either; an optional vLLM server gives fast batched throughput on a GPU.

---

## Two output formats

| Format | Flag | Extension | What you get |
|---|---|---|---|
| **Markdown** (default) | `--format md` | `.md` | Structured document: the **title → `#`**, canonical sections (Abstract, Introduction, Methods, Results, Discussion, …) → `##`, and sub-headings → `###`. |
| **Simple headers** | `--format headers` | `.txt` | The clean full text with every heading kept flat at `##`. Minimal, exactly as the model marked it. |

The top-level vs sub-heading split in `md` mode is **purely deterministic** (a section-name vocabulary, no second LLM call), so output stays reproducible and the install stays light.

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

---

## Install

Requires **Python 3.10+**.

### CPU

```bash
pip install -r requirements.txt          # pulls the CPU build of torch
# or, as a package:
pip install .
```

CPU works out of the box and needs no GPU drivers — but the 8B model is **slow on CPU** (minutes per page). It's fine for a handful of pages or for swapping in a smaller VLM (see *Choosing a model*); for whole papers or batches, use a GPU.

### GPU (recommended)

Install a **CUDA build of PyTorch** first (match your CUDA version — see [pytorch.org](https://pytorch.org/get-started/locally/)), then the rest:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # example: CUDA 12.4
pip install -r requirements.txt
```

A ~16 GB (bf16) GPU comfortably holds Qwen3-VL-8B. The model downloads from Hugging Face on first run and is cached.

### Requirements

| Package | Why |
|---|---|
| `PyMuPDF` | render PDF pages → images |
| `Pillow` | image handling |
| `transformers` (≥4.57) | runs the VLM (`Qwen3VLForConditionalGeneration`) |
| `qwen-vl-utils` | Qwen-VL vision preprocessing |
| `torch` (≥2.4) | model runtime — **install the CUDA build for GPU** |
| `accelerate` | `device_map="auto"` placement |
| `requests` | client for the optional vLLM backend |
| `vllm` *(optional)* | fast batched GPU serving — `pip install ".[vllm]"` |

---

## Usage

### Command line

```bash
# Single PDF → Markdown (auto-selects GPU if available, else CPU)
scientificpub2md paper.pdf

# Simple ## headers instead of structured markdown
scientificpub2md paper.pdf --format headers

# Force CPU / GPU
scientificpub2md paper.pdf --device cpu
scientificpub2md paper.pdf --device cuda

# A whole directory of PDFs into an output folder
scientificpub2md ./pdfs/ -o ./markdown/

# Keep references/acknowledgements/funding (dropped by default)
scientificpub2md paper.pdf --keep-backmatter

# Quick check — just the first 2 pages
scientificpub2md paper.pdf --max-pages 2
```

Output defaults to alongside each PDF (`paper.pdf → paper.md`). Use `-o` for an explicit file (single PDF) or a directory (multiple).

### Python

```python
from scientificpub2md import pdf_to_markdown

md = pdf_to_markdown("paper.pdf")                 # auto CPU/GPU, markdown
headers = pdf_to_markdown("paper.pdf", fmt="headers", device="cpu")
```

Reuse one loaded model across many PDFs:

```python
from scientificpub2md import make_backend, convert_pdf

backend = make_backend("transformers", device="cuda")   # load once
for pdf in pdfs:
    open(pdf.replace(".pdf", ".md"), "w").write(convert_pdf(pdf, backend, fmt="md"))
```

---

## CPU vs GPU

| | `--device cpu` | `--device cuda` | `--backend vllm` |
|---|---|---|---|
| Setup | none | CUDA torch | CUDA torch + `vllm`, run a server |
| Speed | slow (minutes/page) | seconds/page | fastest, **batches pages concurrently** |
| Memory | system RAM | ~16 GB VRAM (8B, bf16) | ~16 GB VRAM + KV cache |
| Use for | a few pages, smaller models | single papers | large batches |

`--device auto` (the default) picks CUDA when a GPU is visible to torch, otherwise CPU.

### Fast batched throughput (optional vLLM backend)

For many papers, serve the model once and let vLLM batch the in-flight page requests:

```bash
pip install ".[vllm]"
./serve_vllm.sh                          # serves Qwen3-VL-8B on :8000 (GPU 0)

# point the client at it; pages are sent concurrently
scientificpub2md ./pdfs/ -o ./md/ --backend vllm --workers 16
```

The server can be remote — set `SCIPUB2MD_VLLM_URL=http://host:8000` (or pass `--vllm-url`).

---

## Choosing a model

The default is `Qwen/Qwen3-VL-8B-Instruct`. Override it with `--model <hf-id>` or the `SCIPUB2MD_VLM_ID` environment variable to use any compatible Qwen3-VL checkpoint (e.g. a smaller variant for CPU, or a larger one for maximum fidelity).

Other environment knobs:

| Variable | Default | Meaning |
|---|---|---|
| `SCIPUB2MD_VLM_ID` | `Qwen/Qwen3-VL-8B-Instruct` | model id |
| `SCIPUB2MD_VLLM_URL` | `http://localhost:8000` | vLLM server URL |
| `SCIPUB2MD_MAX_PIXELS` | `2048·28·28` | per-page resolution budget (transformers backend) |

---

## How it works

1. **Render** — each PDF page → a PNG at `--dpi` (default 170) with PyMuPDF.
2. **Transcribe** — each page image → text with the VLM, guided by one carefully tuned prompt that marks headings as `##`, follows multi-column reading order, drops page furniture and back matter, and emits `SKIP` for all-references pages. Greedy decode → deterministic.
3. **Assemble** — pages concatenated in order (nothing truncated).
4. **Format** — page markers stripped; either kept as flat `##` headers, or restructured into `# / ## / ###` markdown via a deterministic section-name vocabulary.

---

## Notes & limitations

- **OCR is the model's job** — transcription quality tracks the VLM. Dense subscripts/superscripts, unusual glyphs, and complex tables can have errors. Raise `--dpi` if small text is missed.
- **Tables and figures** are transcribed as plain text / captions, not reconstructed as markdown tables or images.
- **Speed on CPU** is the main constraint for whole papers; prefer a GPU or the vLLM backend for batches.
- Determinism assumes a fixed model + prompt + DPI; changing any of them changes the output.

---

## License

MIT — see [LICENSE](LICENSE).

*Born out of the [PubVerse](https://github.com/jimnoneill) extraction pipeline, where this page-by-page VLM method replaced GROBID for building clean full-text scientific corpora.*
