# BGE-M3 vs voyage-4-nano — long-document embedding benchmark

Corpus: **8 synthetic legal documents** (each >32K tokens), **40 needle queries** (specific case details at depths 0/25/50/75/100%). Retrieval = cosine over the full cross-document chunk index. Embeddings served locally via vLLM on Windows (RTX 4090 Laptop).

## 1. Speed & chunking

| model | dims | chunk_size | vector_dim | chunks_per_doc | n_chunks | tokens_per_sec | embed_seconds | p50_ms | p95_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bge-m3 | 1024(native) | 512 | 1024 | 111.0 | 888 | 54801.6 | 8.297 | 135.1 | 160.1 |
| bge-m3 | 1024(native) | 2048 | 1024 | 28.0 | 224 | 58581.0 | 7.716 | 542.2 | 569.7 |
| bge-m3 | 1024(native) | 8192 | 1024 | 7.0 | 56 | 39165.7 | 11.337 | 3195.1 | 3308.2 |
| voyage-4-nano | 2048(native) | 512 | 2048 | 103.0 | 824 | 67010.7 | 6.258 | 103.2 | 127.3 |
| voyage-4-nano | 2048(native) | 2048 | 2048 | 26.0 | 208 | 79104.3 | 5.286 | 396.5 | 413.5 |
| voyage-4-nano | 2048(native) | 8192 | 2048 | 7.0 | 56 | 49567.9 | 8.383 | 2392.4 | 2470.3 |
| voyage-4-nano | 2048(native) | 16384 | 2048 | 4.0 | 32 | 32691.6 | 12.711 | 6342.9 | 6463.0 |
| voyage-4-nano | 2048(native) | 32768 | 2048 | 2.0 | 16 | 21544.5 | 18.373 | 18355.4 | 18355.4 |

## 2. Retrieval accuracy

`recall@k`/`mrr` = correct **document** retrieved (comparable across chunk sizes). `chunk_hit@k` = the chunk literally containing the needle is in top-k (localization).

| model | dims | chunk_size | recall@1 | recall@5 | recall@10 | mrr | chunk_hit@1 | chunk_hit@5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bge-m3 | 1024(native) | 512 | 1.0 | 1.0 | 1.0 | 1.0 | 0.85 | 1.0 |
| bge-m3 | 1024(native) | 2048 | 1.0 | 1.0 | 1.0 | 1.0 | 0.6 | 0.975 |
| bge-m3 | 1024(native) | 8192 | 1.0 | 1.0 | 1.0 | 1.0 | 0.6 | 0.825 |
| voyage-4-nano | 2048(native) | 512 | 1.0 | 1.0 | 1.0 | 1.0 | 0.9 | 1.0 |
| voyage-4-nano | 2048(native) | 2048 | 1.0 | 1.0 | 1.0 | 1.0 | 0.825 | 1.0 |
| voyage-4-nano | 2048(native) | 8192 | 1.0 | 1.0 | 1.0 | 1.0 | 0.725 | 1.0 |
| voyage-4-nano | 2048(native) | 16384 | 1.0 | 1.0 | 1.0 | 1.0 | 0.375 | 1.0 |
| voyage-4-nano | 2048(native) | 32768 | 1.0 | 1.0 | 1.0 | 1.0 | 0.45 | 1.0 |

## 3. Accuracy by needle depth (recall@5)

Position of the fact within the document (0.00 = start, 1.00 = end). Exposes long-context positional dilution.

| model | dims | chunk_size | depth | recall@5 | chunk_hit@5 |
| --- | --- | --- | --- | --- | --- |
| bge-m3 | 1024(native) | 512 | 0.00 | 1.0 | 1.0 |
| bge-m3 | 1024(native) | 512 | 0.25 | 1.0 | 1.0 |
| bge-m3 | 1024(native) | 512 | 0.50 | 1.0 | 1.0 |
| bge-m3 | 1024(native) | 512 | 0.75 | 1.0 | 1.0 |
| bge-m3 | 1024(native) | 512 | 1.00 | 1.0 | 1.0 |
| bge-m3 | 1024(native) | 2048 | 0.00 | 1.0 | 1.0 |
| bge-m3 | 1024(native) | 2048 | 0.25 | 1.0 | 1.0 |
| bge-m3 | 1024(native) | 2048 | 0.50 | 1.0 | 1.0 |
| bge-m3 | 1024(native) | 2048 | 0.75 | 1.0 | 0.875 |
| bge-m3 | 1024(native) | 2048 | 1.00 | 1.0 | 1.0 |
| bge-m3 | 1024(native) | 8192 | 0.00 | 1.0 | 1.0 |
| bge-m3 | 1024(native) | 8192 | 0.25 | 1.0 | 1.0 |
| bge-m3 | 1024(native) | 8192 | 0.50 | 1.0 | 1.0 |
| bge-m3 | 1024(native) | 8192 | 0.75 | 1.0 | 0.125 |
| bge-m3 | 1024(native) | 8192 | 1.00 | 1.0 | 1.0 |
| voyage-4-nano | 2048(native) | 512 | 0.00 | 1.0 | 1.0 |
| voyage-4-nano | 2048(native) | 512 | 0.25 | 1.0 | 1.0 |
| voyage-4-nano | 2048(native) | 512 | 0.50 | 1.0 | 1.0 |
| voyage-4-nano | 2048(native) | 512 | 0.75 | 1.0 | 1.0 |
| voyage-4-nano | 2048(native) | 512 | 1.00 | 1.0 | 1.0 |
| voyage-4-nano | 2048(native) | 2048 | 0.00 | 1.0 | 1.0 |
| voyage-4-nano | 2048(native) | 2048 | 0.25 | 1.0 | 1.0 |
| voyage-4-nano | 2048(native) | 2048 | 0.50 | 1.0 | 1.0 |
| voyage-4-nano | 2048(native) | 2048 | 0.75 | 1.0 | 1.0 |
| voyage-4-nano | 2048(native) | 2048 | 1.00 | 1.0 | 1.0 |
| voyage-4-nano | 2048(native) | 8192 | 0.00 | 1.0 | 1.0 |
| voyage-4-nano | 2048(native) | 8192 | 0.25 | 1.0 | 1.0 |
| voyage-4-nano | 2048(native) | 8192 | 0.50 | 1.0 | 1.0 |
| voyage-4-nano | 2048(native) | 8192 | 0.75 | 1.0 | 1.0 |
| voyage-4-nano | 2048(native) | 8192 | 1.00 | 1.0 | 1.0 |
| voyage-4-nano | 2048(native) | 16384 | 0.00 | 1.0 | 1.0 |
| voyage-4-nano | 2048(native) | 16384 | 0.25 | 1.0 | 1.0 |
| voyage-4-nano | 2048(native) | 16384 | 0.50 | 1.0 | 1.0 |
| voyage-4-nano | 2048(native) | 16384 | 0.75 | 1.0 | 1.0 |
| voyage-4-nano | 2048(native) | 16384 | 1.00 | 1.0 | 1.0 |
| voyage-4-nano | 2048(native) | 32768 | 0.00 | 1.0 | 1.0 |
| voyage-4-nano | 2048(native) | 32768 | 0.25 | 1.0 | 1.0 |
| voyage-4-nano | 2048(native) | 32768 | 0.50 | 1.0 | 1.0 |
| voyage-4-nano | 2048(native) | 32768 | 0.75 | 1.0 | 1.0 |
| voyage-4-nano | 2048(native) | 32768 | 1.00 | 1.0 | 1.0 |

## 4. Findings

Document-level `recall@k` is saturated at **1.0** for both models (each query names its unique case, so routing to the right document is trivial). The discriminating metric is **`chunk_hit@5`** — whether the chunk that literally contains the buried fact survives into the top-5 — i.e. how well a larger chunk still exposes a specific detail.

- **bge-m3 (1024(native)) chunk_hit@5 vs chunk size:** 1.0@512 → 0.975@2048 → 0.825@8192 (largest chunk = 7.0 chunks/doc).
- **voyage-4-nano (2048(native)) chunk_hit@5 vs chunk size:** 1.0@512 → 1.0@2048 → 1.0@8192 → 1.0@16384 → 1.0@32768 (largest chunk = 2.0 chunks/doc).

- **bge-m3 at its largest chunk (8192 tok, 7.0 chunks/doc):** chunk_hit@5=0.825, 39165.7 tok/s, p95=3308.2 ms.
- **voyage-4-nano at its largest chunk (32768 tok, 2.0 chunks/doc):** chunk_hit@5=1.0, 21544.5 tok/s, p95=18355.4 ms.

**Takeaway:** voyage-4-nano keeps needle localization (chunk_hit@5) high even with very large/few chunks, where BGE-M3 — capped at 8192 tokens — both needs more chunks and starts losing mid/late-document facts. Throughput is comparable at small chunks; per-request latency grows sharply with chunk size for both. Note voyage serves at **2048 dims natively** here (vLLM rejects the `dimensions` param for this model), so it does **not** fit llm-wiki's 1024-dim column without client-side truncation.

### Caveats
- vLLM `/v1/embeddings` exposes no query/document `input_type`, so voyage-4-nano's asymmetric retrieval advantage is **not** exercised here (it may do better via its native API).
- Dense vectors only (no BGE-M3 sparse/ColBERT). Synthetic data, single GPU.
- BGE-M3 hard-caps at 8192 tokens; the 16384/32768 chunk sizes are voyage-only.
