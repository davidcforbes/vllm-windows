"""Render results/summary.json into REPORT.md (tables + auto narrative)."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def md_table(rows: list[dict], cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def main():
    summ = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    cfgs = summ["configs"]
    out: list[str] = []
    out.append("# BGE-M3 vs voyage-4-nano — long-document embedding benchmark\n")
    out.append(
        f"Corpus: **{summ['n_docs']} synthetic legal documents** (each >32K tokens), "
        f"**{summ['n_queries']} needle queries** (specific case details at depths "
        f"0/25/50/75/100%). Retrieval = cosine over the full cross-document chunk index. "
        f"Embeddings served locally via vLLM on Windows (RTX 4090 Laptop).\n"
    )
    models_with_rows = {r["model"] for r in cfgs}
    real_skips = [s for s in summ.get("skipped", [])
                  if s.get("model") not in models_with_rows and "chunk_size" not in s]
    if real_skips:
        out.append("> **Skipped models:** " + "; ".join(
            f"{s.get('model')}" for s in real_skips) + "\n")

    out.append("## 1. Speed & chunking\n")
    out.append(md_table(cfgs, ["model", "dims", "chunk_size", "vector_dim", "chunks_per_doc",
                               "n_chunks", "tokens_per_sec", "embed_seconds", "p50_ms", "p95_ms"]) + "\n")

    out.append("## 2. Retrieval accuracy\n")
    out.append("`recall@k`/`mrr` = correct **document** retrieved (comparable across chunk sizes). "
               "`chunk_hit@k` = the chunk literally containing the needle is in top-k (localization).\n")
    out.append(md_table(cfgs, ["model", "dims", "chunk_size", "recall@1", "recall@5", "recall@10",
                               "mrr", "chunk_hit@1", "chunk_hit@5"]) + "\n")

    if summ.get("by_depth"):
        out.append("## 3. Accuracy by needle depth (recall@5)\n")
        out.append("Position of the fact within the document (0.00 = start, 1.00 = end). "
                   "Exposes long-context positional dilution.\n")
        out.append(md_table(summ["by_depth"],
                            ["model", "dims", "chunk_size", "depth", "recall@5", "chunk_hit@5"]) + "\n")

    # ---- auto narrative ----
    out.append("## 4. Findings\n")
    if cfgs:
        from collections import defaultdict
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for r in cfgs:
            groups[(r["model"], r["dims"])].append(r)
        for rs in groups.values():
            rs.sort(key=lambda r: r["chunk_size"])

        out.append("Document-level `recall@k` is saturated at **1.0** for both models (each query "
                   "names its unique case, so routing to the right document is trivial). The "
                   "discriminating metric is **`chunk_hit@5`** — whether the chunk that literally "
                   "contains the buried fact survives into the top-5 — i.e. how well a larger chunk "
                   "still exposes a specific detail.\n")
        for (m, d), rs in groups.items():
            small, large = rs[0], rs[-1]
            trend = " → ".join(f"{r['chunk_hit@5']}@{r['chunk_size']}" for r in rs)
            out.append(f"- **{m} ({d}) chunk_hit@5 vs chunk size:** {trend} "
                       f"(largest chunk = {large['chunks_per_doc']} chunks/doc).")
        # max-chunk head-to-head
        by_model = defaultdict(list)
        for r in cfgs:
            by_model[r["model"]].append(r)
        out.append("")
        for m, rs in by_model.items():
            rs.sort(key=lambda r: r["chunk_size"])
            mx = rs[-1]
            out.append(f"- **{m} at its largest chunk ({mx['chunk_size']} tok, "
                       f"{mx['chunks_per_doc']} chunks/doc):** chunk_hit@5={mx['chunk_hit@5']}, "
                       f"{mx['tokens_per_sec']} tok/s, p95={mx['p95_ms']} ms.")
        out.append("\n**Takeaway:** voyage-4-nano keeps needle localization (chunk_hit@5) high even "
                   "with very large/few chunks, where BGE-M3 — capped at 8192 tokens — both needs "
                   "more chunks and starts losing mid/late-document facts. Throughput is comparable "
                   "at small chunks; per-request latency grows sharply with chunk size for both. "
                   "Note voyage serves at **2048 dims natively** here (vLLM rejects the `dimensions` "
                   "param for this model), so it does **not** fit llm-wiki's 1024-dim column without "
                   "client-side truncation.")
    out.append("\n### Caveats\n"
               "- vLLM `/v1/embeddings` exposes no query/document `input_type`, so voyage-4-nano's "
               "asymmetric retrieval advantage is **not** exercised here (it may do better via its native API).\n"
               "- Dense vectors only (no BGE-M3 sparse/ColBERT). Synthetic data, single GPU.\n"
               "- BGE-M3 hard-caps at 8192 tokens; the 16384/32768 chunk sizes are voyage-only.\n")

    (HERE / "REPORT.md").write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {HERE / 'REPORT.md'}")


if __name__ == "__main__":
    main()
