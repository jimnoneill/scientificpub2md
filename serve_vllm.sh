#!/usr/bin/env bash
# Optional: launch a local vLLM server hosting the VLM, for the fast batched
# `--backend vllm` path. Only useful for large batches on a GPU; the default in-process
# `transformers` backend needs none of this.
#
#   ./serve_vllm.sh                 # serve Qwen3-VL-8B on :8000
#   GPU=1 PORT=8000 ./serve_vllm.sh # pick a GPU / port via env
#
# Then point the client at it:
#   scientificpub2md paper.pdf --backend vllm --workers 16
#   (or set SCIPUB2MD_VLLM_URL=http://host:8000 to reach a remote server)
set -euo pipefail

MODEL="${SCIPUB2MD_VLM_ID:-Qwen/Qwen3-VL-8B-Instruct}"
SERVED_NAME="${SERVED_NAME:-qwen3-vl-8b}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
GPU="${GPU:-0}"
# Fraction of GPU memory vLLM may use. Qwen3-VL-8B is ~16 GB in bf16; raise toward 0.9 on a
# dedicated GPU for a larger KV cache (more concurrent pages), lower it to coexist with others.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

echo "serve_vllm: $MODEL as '$SERVED_NAME' on GPU $GPU, ${HOST}:${PORT} (gpu-mem-util=$GPU_MEM_UTIL)" >&2

exec env CUDA_VISIBLE_DEVICES="$GPU" \
  vllm serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --host "$HOST" \
    --port "$PORT"
