"""Minimal OpenAI-compatible /v1/embeddings client with timing.

Mirrors the request shape used by llm-wiki's embedding-worker
(crates/embedding-worker/src/lib.rs `embed_via_vllm`): JSON {model, input[],
dimensions?}, response `data` sorted by `index`. `dimensions` is sent only when
provided (BGE-M3 rejects it; voyage uses it for matryoshka truncation).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import requests


@dataclass
class EmbedResult:
    vectors: np.ndarray  # [N, dim] float32, L2-normalized
    total_seconds: float
    prompt_tokens: int
    latencies_ms: list[float] = field(default_factory=list)


def _l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


def embed(
    endpoint: str,
    model: str,
    texts: list[str],
    dimensions: int | None = None,
    batch_size: int = 16,
    timeout: float = 600.0,
) -> EmbedResult:
    url = endpoint.rstrip("/") + "/v1/embeddings"
    out: list[list[float]] = [None] * len(texts)  # type: ignore[list-item]
    latencies: list[float] = []
    total_tokens = 0
    t0 = time.perf_counter()
    for base in range(0, len(texts), batch_size):
        batch = texts[base : base + batch_size]
        payload: dict = {"model": model, "input": batch}
        if dimensions is not None:
            payload["dimensions"] = dimensions
        r0 = time.perf_counter()
        resp = requests.post(url, json=payload, timeout=timeout)
        latencies.append((time.perf_counter() - r0) * 1000.0)
        resp.raise_for_status()
        body = resp.json()
        total_tokens += int(body.get("usage", {}).get("prompt_tokens", 0))
        for item in sorted(body["data"], key=lambda d: d["index"]):
            out[base + item["index"]] = item["embedding"]
    total = time.perf_counter() - t0
    mat = _l2_normalize(np.asarray(out, dtype=np.float32))
    return EmbedResult(mat, total, total_tokens, latencies)


def models(endpoint: str, timeout: float = 15.0) -> list[str]:
    r = requests.get(endpoint.rstrip("/") + "/v1/models", timeout=timeout)
    r.raise_for_status()
    return [m["id"] for m in r.json().get("data", [])]
