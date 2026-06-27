"""Page-by-page VLM extraction of a scientific PDF into clean text.

Renders every PDF page to an image (PyMuPDF) and transcribes it ONE PAGE AT A TIME — so
nothing is truncated, however long the paper — then concatenates the pages in order.

Two engines:
  * ``qwen3vl`` (default) — a general VLM (Qwen3-VL-8B) steered by a scientific-manuscript
    prompt: marks headings as ``## ``, drops page furniture + back matter, skips all-reference
    pages. Output is flat-``## `` text the format layer restructures.
  * ``lightonocr`` — LightOnOCR-2-1B, a purpose-built 1B OCR model. Faithful transcription with
    *native* Markdown (headings, tables, LaTeX equations). Smaller/faster; not prompt-steerable,
    so it transcribes everything (no back-matter dropping).

Each engine runs through one of two backends:
  * ``transformers`` — in-process; GPU or CPU. Zero infra.
  * ``vllm`` — a local vLLM server (its ``/v1/chat/completions`` HTTP API); pages can be sent
    concurrently for throughput. Runs entirely on your own machine — no external service.

Decoding is greedy / temperature 0, so re-extracting a PDF is byte-reproducible.
"""
from __future__ import annotations

import io
import os
import time

from .prompts import page_prompt

DEFAULT_MODEL = os.environ.get("SCIPUB2MD_VLM_ID", "Qwen/Qwen3-VL-8B-Instruct")
LIGHTONOCR_MODEL = os.environ.get("SCIPUB2MD_LIGHTONOCR_ID", "lightonai/LightOnOCR-2-1B")
DEFAULT_DPI = 170
# Qwen-VL dynamic-resolution budget: a generous max keeps dense scientific text legible.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = int(os.environ.get("SCIPUB2MD_MAX_PIXELS", str(2048 * 28 * 28)))

PAGE_MARKER = "<<<PAGE {n}>>>"


# --------------------------------------------------------------------------------------
# PDF -> page images
# --------------------------------------------------------------------------------------
def render_pages(pdf_path, dpi=DEFAULT_DPI, max_pages=None):
    """Yield ``(page_no, total_pages, PIL.Image)`` for each page via PyMuPDF."""
    import fitz  # PyMuPDF
    from PIL import Image

    doc = fitz.open(pdf_path)
    try:
        n = len(doc) if max_pages is None else min(len(doc), max_pages)
        for i in range(n):
            png = doc[i].get_pixmap(dpi=dpi).tobytes("png")
            yield i + 1, len(doc), Image.open(io.BytesIO(png)).convert("RGB")
    finally:
        doc.close()


def _resize_longest(img, longest):
    """Downscale so the longest side is <= ``longest`` px (aspect preserved); never upscales."""
    if not longest or max(img.size) <= longest:
        return img
    from PIL import Image

    resample = getattr(Image, "Resampling", Image).LANCZOS
    out = img.copy()
    out.thumbnail((longest, longest), resample)
    return out


def _is_skip(text: str) -> bool:
    return text.upper().strip(" .") == "SKIP"


def _resolve_device(device, *, allow_mps=False):
    """Resolve 'auto' to a concrete device and validate an explicit request against torch."""
    import torch

    if device == "auto":
        if allow_mps and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but no CUDA GPU is visible to torch.")
    return device


import contextlib


@contextlib.contextmanager
def _left_padding(processor):
    """Temporarily set the processor's tokenizer to left-pad — required for correct decoder-only
    batched generation (so the generated tokens of every sequence in the batch stay aligned)."""
    tok = getattr(processor, "tokenizer", None)
    if tok is None or getattr(tok, "padding_side", None) == "left":
        yield
        return
    prev = tok.padding_side
    tok.padding_side = "left"
    try:
        yield
    finally:
        tok.padding_side = prev


# --------------------------------------------------------------------------------------
# Qwen3-VL via transformers (in-process; CPU or GPU)
# --------------------------------------------------------------------------------------
class TransformersBackend:
    """In-process Qwen3-VL via 🤗 transformers. Loads once, then transcribes page images.

    device: ``"auto"`` (CUDA if available, else CPU), ``"cuda"``, or ``"cpu"``. On GPU the weights
    load in bfloat16 (~16 GB for the 8B); on CPU in float32 (correct but slow — minutes per page).
    """

    native_markdown = False
    default_dpi = DEFAULT_DPI

    def __init__(self, model_id=DEFAULT_MODEL, device="auto", keep_backmatter=False):
        import torch

        self.model_id = model_id
        self.keep_backmatter = keep_backmatter
        self._model = None
        self._processor = None
        self.device = _resolve_device(device, allow_mps=True)
        # bf16 on CUDA; fp16 on Apple MPS (bf16 is poorly supported there); fp32 on CPU.
        self._dtype = (
            torch.bfloat16 if self.device == "cuda"
            else torch.float16 if self.device == "mps"
            else torch.float32
        )

    _TV_HELP = (
        "The qwen3vl engine's transformers backend needs torchvision (the Qwen3-VL processor "
        "requires it). Install with `pip install 'scientificpub2md[qwen]'` (install torch + "
        "torchvision from the SAME index so their builds match), or use `--engine lightonocr` "
        "(no torchvision), or `--backend vllm`."
    )

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        t0 = time.time()
        print(f"  loading {self.model_id} on {self.device} ({self._dtype})…", flush=True)
        # Load the processor first — it pulls in torchvision, so a missing-torchvision install
        # fails fast here (with a clear message) instead of after downloading 16 GB of weights.
        try:
            self._processor = AutoProcessor.from_pretrained(
                self.model_id, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
            )
        except ImportError as e:
            if "torchvision" in str(e).lower():
                raise ImportError(self._TV_HELP) from e
            raise
        if self.device == "cuda":
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype=self._dtype, device_map="auto"
            )
        else:  # cpu / mps: load then move (device_map='auto' doesn't target MPS well)
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype=self._dtype
            ).to(self.device)
        self._model.eval()
        print(f"  loaded in {time.time() - t0:.0f}s", flush=True)

    def transcribe(self, pil_img, max_new_tokens=6144):
        import torch

        self._load()
        # Native processor path: pass the in-memory PIL page inline and let the chat template do the
        # vision preprocessing. Avoids qwen_vl_utils (and its torchvision dependency) entirely.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": page_prompt(self.keep_backmatter)},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self._model.device)
        with torch.no_grad():
            gen = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = gen[:, inputs["input_ids"].shape[1]:]
        out = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        return "" if _is_skip(out) else out

    def transcribe_batch(self, pil_imgs, max_new_tokens=6144):
        """Transcribe several pages in one padded ``generate`` call (throughput win on GPU).

        Greedy + a correct attention mask makes this equivalent to per-page ``transcribe``;
        left-padding keeps decoder-only generation aligned across the batch."""
        import torch

        if len(pil_imgs) == 1:
            return [self.transcribe(pil_imgs[0], max_new_tokens=max_new_tokens)]
        self._load()
        prompt = page_prompt(self.keep_backmatter)
        conversations = [
            [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": prompt}]}]
            for im in pil_imgs
        ]
        with _left_padding(self._processor):
            inputs = self._processor.apply_chat_template(
                conversations, add_generation_prompt=True, tokenize=True, return_dict=True,
                return_tensors="pt", padding=True,
            ).to(self._model.device)
        with torch.no_grad():
            gen = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = gen[:, inputs["input_ids"].shape[1]:]
        outs = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return ["" if _is_skip(o.strip()) else o.strip() for o in outs]


# --------------------------------------------------------------------------------------
# LightOnOCR-2-1B via transformers (in-process; CPU, CUDA, or MPS)
# --------------------------------------------------------------------------------------
class LightOnOCRBackend:
    """In-process LightOnOCR-2-1B via 🤗 transformers — a purpose-built 1B OCR model.

    Faithful, prompt-free transcription (image-only chat turn) producing native Markdown with
    tables and LaTeX equations. ~2-3 GB in bf16, so it runs comfortably on a small GPU, and on
    CPU/MPS far faster than the 8B Qwen path. Pages are rendered at 200 DPI and downscaled to a
    1540 px longest side (the model card's recommendation).

    NOTE: the model classes were upstreamed to transformers after 4.57; on older versions this
    falls back to ``AutoModelForImageTextToText`` (+ ``trust_remote_code``). ``keep_backmatter``
    is accepted for interface parity but has no effect — LightOnOCR is not prompt-steerable and
    transcribes the whole page (references included).
    """

    native_markdown = True
    default_dpi = 200

    def __init__(self, model_id=LIGHTONOCR_MODEL, device="auto", keep_backmatter=False,
                 resize_longest=1540, trust_remote_code=True):
        import torch

        self.model_id = model_id
        self.resize_longest = resize_longest
        self.trust_remote_code = trust_remote_code
        self._model = None
        self._processor = None
        self.device = _resolve_device(device, allow_mps=True)
        # bf16 on CUDA; float32 on CPU/MPS (bf16 is poorly supported there).
        self._dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

    def _load(self):
        if self._model is not None:
            return
        try:  # upstreamed in transformers (post-4.57)
            from transformers import LightOnOcrForConditionalGeneration as _Model
            from transformers import LightOnOcrProcessor as _Proc

            model_kwargs, proc_kwargs = {}, {}
        except ImportError:  # older transformers: resolve via Auto* + remote code
            from transformers import AutoModelForImageTextToText as _Model
            from transformers import AutoProcessor as _Proc

            model_kwargs = {"trust_remote_code": self.trust_remote_code}
            proc_kwargs = {"trust_remote_code": self.trust_remote_code}

        t0 = time.time()
        print(f"  loading {self.model_id} on {self.device} ({self._dtype})…", flush=True)
        self._model = _Model.from_pretrained(self.model_id, torch_dtype=self._dtype, **model_kwargs).to(self.device)
        self._model.eval()
        self._processor = _Proc.from_pretrained(self.model_id, **proc_kwargs)
        print(f"  loaded in {time.time() - t0:.0f}s", flush=True)

    def transcribe(self, pil_img, max_new_tokens=4096):
        import torch

        self._load()
        img = _resize_longest(pil_img, self.resize_longest)
        # Image-only turn (no text instruction) — the model card's documented usage.
        conversation = [{"role": "user", "content": [{"type": "image", "image": img}]}]
        inputs = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        inputs = {
            k: (v.to(device=self.device, dtype=self._dtype) if v.is_floating_point() else v.to(self.device))
            for k, v in inputs.items()
        }
        with torch.no_grad():
            out_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out_ids[0, inputs["input_ids"].shape[1]:]
        return self._processor.decode(gen, skip_special_tokens=True).strip()

    def transcribe_batch(self, pil_imgs, max_new_tokens=4096):
        """Transcribe several pages in one padded ``generate`` call (throughput win on GPU).

        Greedy + a correct attention mask makes this byte-identical to calling ``transcribe`` per
        page; left-padding is used so decoder-only generation stays aligned across the batch."""
        import torch

        if len(pil_imgs) == 1:
            return [self.transcribe(pil_imgs[0], max_new_tokens=max_new_tokens)]
        self._load()
        imgs = [_resize_longest(im, self.resize_longest) for im in pil_imgs]
        conversations = [[{"role": "user", "content": [{"type": "image", "image": im}]}] for im in imgs]
        with _left_padding(self._processor):
            inputs = self._processor.apply_chat_template(
                conversations, add_generation_prompt=True, tokenize=True, return_dict=True,
                return_tensors="pt", padding=True,
            )
        inputs = {
            k: (v.to(device=self.device, dtype=self._dtype) if v.is_floating_point() else v.to(self.device))
            for k, v in inputs.items()
        }
        with torch.no_grad():
            out_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out_ids[:, inputs["input_ids"].shape[1]:]
        return [t.strip() for t in self._processor.batch_decode(gen, skip_special_tokens=True)]


# --------------------------------------------------------------------------------------
# vLLM backend (local vLLM HTTP server; GPU; supports concurrent pages) — both engines
# --------------------------------------------------------------------------------------
class VLLMBackend:
    """Client for a local vLLM server hosting a VLM/OCR model.

    Talks to vLLM's ``/v1/chat/completions`` HTTP endpoint with plain ``requests`` — that URL
    schema is a widely-adopted wire format (vLLM, Ollama, llama.cpp, TGI, ... all implement it).
    Nothing here calls any external/hosted service; the server runs on your own machine/tailnet.

    Works for either engine: pass ``prompt`` (the per-page instruction) for the Qwen3-VL path, or
    leave it ``None`` for LightOnOCR's image-only path. Deterministic via temperature 0 + a fixed
    seed. If a dense page is truncated at the token ceiling (finish_reason == 'length') the page is
    retried once at double the budget and the longer transcription is kept.
    """

    def __init__(self, url=None, model="qwen3-vl-8b", prompt=None, *, native_markdown=False,
                 default_dpi=DEFAULT_DPI, resize_longest=None, temperature=0.0, use_skip=False):
        self.url = (url or os.environ.get("SCIPUB2MD_VLLM_URL", "http://localhost:8000")).rstrip("/")
        self.model = model
        self.prompt = prompt
        self.native_markdown = native_markdown
        self.default_dpi = default_dpi
        self.resize_longest = resize_longest
        self.temperature = temperature
        self.use_skip = use_skip  # honor the 'SKIP' sentinel (Qwen prompt) — LightOnOCR never emits it
        self.device = "cuda (vllm server)"

    def transcribe(self, pil_img, max_new_tokens=6144):
        import base64

        import requests

        img = _resize_longest(pil_img, self.resize_longest)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]
        if self.prompt:
            content.append({"type": "text", "text": self.prompt})

        def _call(budget):
            payload = {
                "model": self.model,
                "temperature": self.temperature,
                "seed": 42,
                "max_tokens": budget,
                "min_tokens": 16,
                "messages": [{"role": "user", "content": content}],
            }
            r = requests.post(self.url + "/v1/chat/completions", json=payload, timeout=600)
            r.raise_for_status()
            ch = r.json()["choices"][0]
            return (ch["message"].get("content") or "").strip(), ch.get("finish_reason")

        out, reason = _call(max_new_tokens)
        if reason == "length":
            out2, _ = _call(max_new_tokens * 2)
            if len(out2) > len(out):
                out = out2
        if self.use_skip and _is_skip(out):
            return ""
        return out


# --------------------------------------------------------------------------------------
# factory
# --------------------------------------------------------------------------------------
def make_backend(backend="transformers", *, engine="lightonocr", device="auto", model=None,
                 keep_backmatter=False, vllm_url=None, vllm_model=None):
    """Construct a backend for the requested ``engine`` ('lightonocr' or 'qwen3vl') and ``backend``
    runtime ('transformers' or 'vllm')."""
    if engine not in ("qwen3vl", "lightonocr"):
        raise ValueError(f"unknown engine {engine!r} (expected 'qwen3vl' or 'lightonocr')")

    if backend == "transformers":
        if engine == "lightonocr":
            return LightOnOCRBackend(model_id=model or LIGHTONOCR_MODEL, device=device,
                                     keep_backmatter=keep_backmatter)
        return TransformersBackend(model_id=model or DEFAULT_MODEL, device=device,
                                   keep_backmatter=keep_backmatter)

    if backend == "vllm":
        if engine == "lightonocr":
            return VLLMBackend(url=vllm_url, model=vllm_model or LIGHTONOCR_MODEL, prompt=None,
                               native_markdown=True, default_dpi=200, resize_longest=1540)
        return VLLMBackend(url=vllm_url, model=vllm_model or "qwen3-vl-8b",
                           prompt=page_prompt(keep_backmatter), native_markdown=False,
                           default_dpi=DEFAULT_DPI, resize_longest=None, use_skip=True)

    raise ValueError(f"unknown backend {backend!r} (expected 'transformers' or 'vllm')")


# --------------------------------------------------------------------------------------
# whole-PDF drivers
# --------------------------------------------------------------------------------------
def _dpi_for(backend, dpi):
    return dpi if dpi is not None else getattr(backend, "default_dpi", DEFAULT_DPI)


def extract_pdf(pdf_path, backend, dpi=None, max_pages=None, verbose=True, batch_size=1):
    """Render + transcribe every page; concatenate with page markers. No truncation.

    batch_size > 1 transcribes pages in padded batches via ``backend.transcribe_batch`` (a GPU
    throughput win; greedy decoding keeps it equivalent to the per-page path). Backends without a
    ``transcribe_batch`` fall back to per-page transcription.
    """
    dpi = _dpi_for(backend, dpi)
    can_batch = batch_size and batch_size > 1 and hasattr(backend, "transcribe_batch")
    parts = []
    t0 = time.time()
    batch = []  # list of (pno, total, img) awaiting a batched transcribe

    def _flush():
        imgs = [img for _pno, _total, img in batch]
        texts = backend.transcribe_batch(imgs) if can_batch else [backend.transcribe(imgs[0])]
        for (pno, total, _img), txt in zip(batch, texts):
            parts.append(f"\n\n{PAGE_MARKER.format(n=pno)}\n{txt}")
            if verbose:
                print(f"    page {pno}/{total}: {len(txt)} chars ({time.time() - t0:.0f}s)", flush=True)
        batch.clear()

    for pno, total, img in render_pages(pdf_path, dpi=dpi, max_pages=max_pages):
        if not can_batch:
            txt = backend.transcribe(img)
            parts.append(f"\n\n{PAGE_MARKER.format(n=pno)}\n{txt}")
            if verbose:
                print(f"    page {pno}/{total}: {len(txt)} chars ({time.time() - t0:.0f}s)", flush=True)
            continue
        batch.append((pno, total, img))
        if len(batch) >= batch_size:
            _flush()
    if batch:
        _flush()
    return "".join(parts).strip()


def extract_pdf_concurrent(pdf_path, backend, dpi=None, max_pages=None, workers=8, verbose=True,
                           batch_size=1):
    """Transcribe pages concurrently (vLLM backend) and reassemble in page order.

    Only meaningful for the vLLM backend, whose server batches the in-flight requests. Falls back
    to (optionally batched) in-process extraction for the transformers backends — a single loaded
    model is not thread-safe to call concurrently, but ``batch_size`` batches pages in one forward.
    """
    if not isinstance(backend, VLLMBackend):
        return extract_pdf(pdf_path, backend, dpi=dpi, max_pages=max_pages, verbose=verbose,
                           batch_size=batch_size)
    import concurrent.futures as cf

    dpi = _dpi_for(backend, dpi)
    pages = list(render_pages(pdf_path, dpi=dpi, max_pages=max_pages))
    out = {}
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(backend.transcribe, img): pno for pno, _, img in pages}
        for fut in cf.as_completed(futs):
            pno = futs[fut]
            try:
                out[pno] = fut.result()
            except Exception as e:  # one bad page shouldn't sink the paper
                out[pno] = ""
                print(f"    [page {pno} failed: {e}]", flush=True)
    if verbose:
        print(f"    {len(pages)} pages in {time.time() - t0:.0f}s", flush=True)
    return "".join(f"\n\n{PAGE_MARKER.format(n=p)}\n{out[p]}" for p in sorted(out)).strip()
