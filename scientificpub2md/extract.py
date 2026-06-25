"""Page-by-page VLM extraction of a scientific PDF into clean text with ``## `` headers.

Renders every PDF page to an image (PyMuPDF) and transcribes it ONE PAGE AT A TIME with a
vision-language model (Qwen3-VL-8B by default) — so nothing is truncated, however long the
paper. Decoding is greedy (deterministic), so re-extracting a PDF is byte-reproducible.

Two backends:
  * ``transformers`` (default) — loads the model in-process; runs on GPU **or** CPU. Zero infra.
  * ``vllm`` — talks to an OpenAI-compatible vLLM server (GPU); pages can be sent concurrently
    for much higher throughput on large batches. See ``serve_vllm.sh``.
"""
from __future__ import annotations

import io
import os
import time

from .prompts import page_prompt

DEFAULT_MODEL = os.environ.get("SCIPUB2MD_VLM_ID", "Qwen/Qwen3-VL-8B-Instruct")
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


def _is_skip(text: str) -> bool:
    return text.upper().strip(" .") == "SKIP"


# --------------------------------------------------------------------------------------
# transformers backend (in-process; CPU or GPU)
# --------------------------------------------------------------------------------------
class TransformersBackend:
    """In-process Qwen3-VL via 🤗 transformers. Loads once, then transcribes page images.

    device: ``"auto"`` (CUDA if available, else CPU), ``"cuda"``, or ``"cpu"``.
    On GPU the weights load in bfloat16 (~16 GB for the 8B); on CPU they load in float32
    (correct but slow — minutes per page — best for a handful of pages or a smaller model).
    """

    def __init__(self, model_id=DEFAULT_MODEL, device="auto", keep_backmatter=False):
        self.model_id = model_id
        self.keep_backmatter = keep_backmatter
        self._model = None
        self._processor = None
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but no CUDA GPU is visible to torch.")
        self.device = device
        self._dtype = torch.bfloat16 if device == "cuda" else torch.float32

    def _load(self):
        if self._model is not None:
            return
        import torch  # noqa: F401
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        t0 = time.time()
        print(f"  loading {self.model_id} on {self.device} ({self._dtype})…", flush=True)
        device_map = "auto" if self.device == "cuda" else "cpu"
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=self._dtype, device_map=device_map
        )
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(
            self.model_id, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
        )
        print(f"  loaded in {time.time() - t0:.0f}s", flush=True)

    def transcribe(self, pil_img, max_new_tokens=6144):
        import torch
        from qwen_vl_utils import process_vision_info

        self._load()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": page_prompt(self.keep_backmatter)},
                ],
            }
        ]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = self._processor(
            text=[text], images=image_inputs, padding=True, return_tensors="pt"
        ).to(self._model.device)
        with torch.no_grad():
            gen = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [g[len(i):] for i, g in zip(inputs.input_ids, gen)]
        out = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        return "" if _is_skip(out) else out


# --------------------------------------------------------------------------------------
# vLLM backend (OpenAI-compatible server; GPU; supports concurrent pages)
# --------------------------------------------------------------------------------------
class VLLMBackend:
    """Client for an OpenAI-compatible vLLM server hosting the VLM.

    Deterministic via temperature 0 + a fixed seed. If a dense page is truncated at the token
    ceiling (finish_reason == 'length') the page is retried once at double the budget and the
    longer transcription is kept.
    """

    def __init__(self, url=None, model="qwen3-vl-8b", keep_backmatter=False):
        self.url = (url or os.environ.get("SCIPUB2MD_VLLM_URL", "http://localhost:8000")).rstrip("/")
        self.model = model
        self.keep_backmatter = keep_backmatter
        self.device = "cuda (vllm server)"

    def transcribe(self, pil_img, max_new_tokens=6144):
        import base64

        import requests

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        prompt = page_prompt(self.keep_backmatter)

        def _call(budget):
            payload = {
                "model": self.model,
                "temperature": 0.0,
                "seed": 42,
                "max_tokens": budget,
                "min_tokens": 16,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
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
        return "" if _is_skip(out) else out


def make_backend(backend="transformers", *, device="auto", model=None, keep_backmatter=False,
                 vllm_url=None, vllm_model="qwen3-vl-8b"):
    """Construct the requested backend."""
    if backend == "vllm":
        return VLLMBackend(url=vllm_url, model=vllm_model, keep_backmatter=keep_backmatter)
    if backend == "transformers":
        return TransformersBackend(
            model_id=model or DEFAULT_MODEL, device=device, keep_backmatter=keep_backmatter
        )
    raise ValueError(f"unknown backend {backend!r} (expected 'transformers' or 'vllm')")


# --------------------------------------------------------------------------------------
# whole-PDF drivers
# --------------------------------------------------------------------------------------
def extract_pdf(pdf_path, backend, dpi=DEFAULT_DPI, max_pages=None, verbose=True):
    """Render + transcribe every page sequentially; concatenate with page markers. No truncation."""
    parts = []
    t0 = time.time()
    for pno, total, img in render_pages(pdf_path, dpi=dpi, max_pages=max_pages):
        txt = backend.transcribe(img)
        parts.append(f"\n\n{PAGE_MARKER.format(n=pno)}\n{txt}")
        if verbose:
            print(f"    page {pno}/{total}: {len(txt)} chars ({time.time() - t0:.0f}s)", flush=True)
    return "".join(parts).strip()


def extract_pdf_concurrent(pdf_path, backend, dpi=DEFAULT_DPI, max_pages=None, workers=8, verbose=True):
    """Transcribe pages concurrently (vLLM backend) and reassemble in page order.

    Only meaningful for the vLLM backend, whose server batches the in-flight requests. Falls back
    to sequential extraction for the transformers backend (a single in-process model is not
    thread-safe to call concurrently).
    """
    if not isinstance(backend, VLLMBackend):
        return extract_pdf(pdf_path, backend, dpi=dpi, max_pages=max_pages, verbose=verbose)
    import concurrent.futures as cf

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
