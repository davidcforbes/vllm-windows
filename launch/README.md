# Native Windows vLLM launch scripts

Clean, Docker-free launchers for the three models, all served by the native
Windows vLLM build (`C:\dev\vllm-windows\.venv`). Each serves in the foreground
(Ctrl-C to stop) and bakes in the Windows essentials: CUDA 13.3 on PATH,
`VLLM_USE_FLASHINFER_SAMPLER=0` for generative models (FlashInfer's sampler JIT
fails on Windows), and running from `%TEMP%` to avoid source-shadowing the
installed `vllm._C`.

| # | Model | Script | Endpoint | Served name |
|---|-------|--------|----------|-------------|
| 1 | Chandra OCR 2 (VLM) | `C:\dev\chandra\scripts\serve_vllm.ps1` | `:8000/v1` | `chandra` |
| 2 | BGE-M3 (embeddings) | `launch\serve_bge_m3.ps1` | `:8001/v1/embeddings` | `bge-m3` |
| 3 | Gemma 12B 4-bit AWQ (LLM) | `launch\serve_gemma.ps1` | `:8002/v1/chat/completions` | `gemma` |

```powershell
# 1. Chandra OCR (PDF/image -> markdown). Serve-only launcher. See c:\dev\chandra.
pwsh C:\dev\chandra\scripts\serve_vllm.ps1

# 2. BGE-M3 embeddings
pwsh .\launch\serve_bge_m3.ps1

# 3. Gemma 12B AWQ — REQUIRES -Model (no verified official "Gemma 4 12B" AWQ;
#    point at a real AWQ/GPTQ build or local path)
pwsh .\launch\serve_gemma.ps1 -Model <org/gemma-12b-it-AWQ-or-local-path>
```

> **Chandra has three entry points (in chandra/Book-Scan, not here), all on the
> same `:8000` / `chandra` endpoint:**
> - `C:\dev\chandra\scripts\serve_vllm.ps1` — serve-only (canonical; what
>   llm-wiki's ingest worker and the `chandra` CLI target).
> - `C:\dev\chandra\scripts\run_pipeline_native.ps1` — serve **+** run the full
>   `chandra` CLI pipeline.
> - `C:\dev\Book-Scan\scripts\serve_vllm_native.ps1` — serve **+** optional
>   Book-Scan `ocr-folder` batch via `-PagesDir`/`-OutFile`.

## VRAM note (16 GB RTX 4090 Mobile)

- **BGE-M3** is tiny (~2 GB) and can share the GPU.
- **Chandra** (~15 GB) and **Gemma 12B** (~7 GB weights + KV) each want most of
  the card — run them **one at a time**, not together. Keep the embedder on CPU
  (ONNX, per llm-wiki ADR 0028) or BGE-M3's small footprint if you need
  embeddings alongside one of them.
- Each script's defaults are tuned for 16 GB; pass `-GpuMemUtil` / `-MaxModelLen`
  (and `-Cudagraphs` on Gemma, `-EnforceEager` to save VRAM) to adjust.
