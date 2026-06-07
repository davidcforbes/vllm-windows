# Embedding comparison: BGE-M3 vs voyage-4-nano (long legal documents)

Benchmarks two embedding models served locally by vLLM (Windows) on **speed** and
**retrieval of specific facts** buried in long (>32K-token) synthetic legal documents,
as a function of chunk size. See [`REPORT.md`](REPORT.md) for results.

Why it matters: BGE-M3 caps at **8,192** input tokens; voyage-4-nano handles **32,768**.
A >32K-token document therefore needs ≥4–5 BGE chunks but only 1–2 voyage chunks — this
harness measures whether voyage's "fewer/larger chunks" still find specific details, and how
fast.

## Requirements

`pip install -r requirements.txt` (numpy, requests, transformers). Uses the repo's vLLM venv.

## Run

The two models are served **one at a time** (vLLM's data-parallel rendezvous binds the host's
default RPC port, so concurrent same-host embedding servers collide on Windows). Each server
must be launched from **outside** the repo dir — the serve `.bat` files handle this.

```bat
:: 1) generate the corpus (deterministic; asserts each doc > 32768 tokens)
..\..\.venv\Scripts\python.exe gen_data.py

:: 2) BGE-M3 pass
start serve_bge.bat            :: waits, serves on http://127.0.0.1:8001
..\..\.venv\Scripts\python.exe run_bench.py --merge
:: stop the BGE server, then wait ~2 min for its RPC port to leave TIME_WAIT

:: 3) voyage-4-nano pass
start serve_voyage.bat         :: serves on http://127.0.0.1:8003
..\..\.venv\Scripts\python.exe run_bench.py --merge

:: 4) render the report
..\..\.venv\Scripts\python.exe make_report.py
```

`run_bench.py --merge` only benchmarks whichever endpoint is up and merges into existing
`results/`, so running the two passes sequentially accumulates both models.

## Files

- `gen_data.py` — deterministic synthetic legal docs + injected needle facts → `data/`.
- `chunking.py` — per-model token chunker (sliding window + overlap).
- `embed_client.py` — OpenAI `/v1/embeddings` client with timing.
- `run_bench.py` — chunk → embed → score retrieval; writes `results/`.
- `make_report.py` — renders `REPORT.md`.
- `serve_bge.bat` / `serve_voyage.bat` — vLLM embedding servers.

## Notes / gotchas (Windows)

- vLLM serves voyage-4-nano at its **native 2048 dims**; it rejects the OpenAI `dimensions`
  param ("does not support matryoshka"), so it cannot be truncated to 1024 server-side.
- `/v1/embeddings` has no `input_type`, so voyage's asymmetric query/document advantage is not used.
- Serve scripts run from `%TEMP%` to avoid the source-shadowing of the installed `vllm._C`.
