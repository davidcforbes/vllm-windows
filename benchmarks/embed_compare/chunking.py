"""Token-based chunking using each model's own tokenizer.

Chunks are produced by a sliding window over the model's token ids and decoded
back to text (the embedding server re-tokenizes). A small margin is left under
the model's max so special tokens don't overflow the context.
"""

from __future__ import annotations

from functools import lru_cache

SPECIAL_MARGIN = 16


@lru_cache(maxsize=4)
def get_tokenizer(hf_id: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)


def token_count(hf_id: str, text: str) -> int:
    tok = get_tokenizer(hf_id)
    return len(tok(text, add_special_tokens=False)["input_ids"])


def chunk_text(
    hf_id: str, text: str, max_tokens: int, model_max: int, overlap_frac: float = 0.15
) -> list[str]:
    """Split `text` into decoded chunks of <= effective token length.

    effective = min(max_tokens, model_max - SPECIAL_MARGIN). Returns at least one
    chunk. Stride = effective * (1 - overlap_frac).
    """
    tok = get_tokenizer(hf_id)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    eff = max(16, min(max_tokens, model_max - SPECIAL_MARGIN))
    if len(ids) <= eff:
        return [text]
    stride = max(1, int(eff * (1.0 - overlap_frac)))
    chunks: list[str] = []
    start = 0
    while start < len(ids):
        window = ids[start : start + eff]
        chunks.append(tok.decode(window, skip_special_tokens=True))
        if start + eff >= len(ids):
            break
        start += stride
    return chunks
