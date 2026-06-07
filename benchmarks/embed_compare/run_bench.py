"""Run the BGE-M3 vs voyage-4-nano embedding benchmark.

For each (model, dims, chunk_size): chunk all docs with the model's tokenizer,
embed the chunks (timed) and the gold queries, then score retrieval of each
needle's query against the full cross-document chunk index.

Metrics per config:
  speed   - embed wall-time, prompt tokens/sec, chunks/doc, p50/p95 request latency
  doc     - recall@{1,5,10} + MRR (correct DOC retrieved; comparable across chunk sizes)
  chunk   - hit@{1,5} (the chunk literally containing the needle is in top-k)
  + a per-needle-depth breakdown (positional degradation in long contexts)

Writes results/by_config.csv, results/by_depth.csv, results/summary.json.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import chunking
import embed_client

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DOCS = DATA / "docs"
RESULTS = HERE / "results"
KS = [1, 5, 10]

CONFIGS = [
    {"model": "bge-m3", "endpoint": "http://127.0.0.1:8001", "tokenizer": "BAAI/bge-m3",
     "model_max": 8192, "dims": [None], "chunk_sizes": [512, 2048, 8192]},
    # vLLM serves voyage-4-nano at its NATIVE dim (2048) only; it rejects the
    # OpenAI `dimensions` param ("does not support matryoshka"), so 1024/2048
    # truncation is not available server-side. dims=None -> no dimensions field.
    {"model": "voyage-4-nano", "endpoint": "http://127.0.0.1:8003", "tokenizer": "voyageai/voyage-4-nano",
     "model_max": 32768, "dims": [None], "chunk_sizes": [512, 2048, 8192, 16384, 32768]},
]


def load_corpus():
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    docs = {d["doc_id"]: (DOCS / f"{d['doc_id']}.txt").read_text(encoding="utf-8") for d in manifest}
    queries = []
    for d in manifest:
        for nd in d["needles"]:
            queries.append({"needle_id": nd["needle_id"], "doc_id": d["doc_id"],
                            "query": nd["query"], "answer": nd["answer"], "depth": nd["depth"]})
    return docs, queries


def run_one(cfg, dims, chunk_size, docs, queries):
    tok = cfg["tokenizer"]
    chunk_texts, chunk_doc = [], []
    for doc_id, text in docs.items():
        for ch in chunking.chunk_text(tok, text, chunk_size, cfg["model_max"]):
            chunk_texts.append(ch)
            chunk_doc.append(doc_id)

    emb = embed_client.embed(cfg["endpoint"], cfg["model"], chunk_texts, dimensions=dims)
    qemb = embed_client.embed(cfg["endpoint"], cfg["model"], [q["query"] for q in queries], dimensions=dims)
    sims = qemb.vectors @ emb.vectors.T  # [Q, C]
    chunk_doc_arr = np.asarray(chunk_doc)

    recall = {k: 0 for k in KS}
    chunk_hit = {k: 0 for k in KS}
    mrr = 0.0
    by_depth: dict[str, dict] = {}
    for qi, q in enumerate(queries):
        order = np.argsort(-sims[qi])
        ranked_docs = chunk_doc_arr[order]
        correct = ranked_docs == q["doc_id"]
        first_doc = int(np.argmax(correct)) if correct.any() else None
        contains = np.fromiter((q["answer"] in chunk_texts[ci] for ci in order), dtype=bool, count=len(order))
        first_chunk = int(np.argmax(contains)) if contains.any() else None
        for k in KS:
            if first_doc is not None and first_doc < k:
                recall[k] += 1
            if first_chunk is not None and first_chunk < k:
                chunk_hit[k] += 1
        if first_doc is not None:
            mrr += 1.0 / (first_doc + 1)
        bd = by_depth.setdefault(f"{q['depth']:.2f}", {"r5": 0, "c5": 0, "n": 0})
        bd["n"] += 1
        bd["r5"] += int(first_doc is not None and first_doc < 5)
        bd["c5"] += int(first_chunk is not None and first_chunk < 5)

    q_n = len(queries)
    native = int(emb.vectors.shape[1])
    dims_label = str(dims) if dims is not None else f"{native}(native)"
    row = {
        "model": cfg["model"], "dims": dims_label, "chunk_size": chunk_size,
        "vector_dim": native,
        "n_chunks": len(chunk_texts), "chunks_per_doc": round(len(chunk_texts) / len(docs), 2),
        "embed_seconds": round(emb.total_seconds, 3), "prompt_tokens": emb.prompt_tokens,
        "tokens_per_sec": round(emb.prompt_tokens / emb.total_seconds, 1) if emb.total_seconds else 0.0,
        "p50_ms": round(float(np.percentile(emb.latencies_ms, 50)), 1),
        "p95_ms": round(float(np.percentile(emb.latencies_ms, 95)), 1),
        "recall@1": round(recall[1] / q_n, 3), "recall@5": round(recall[5] / q_n, 3),
        "recall@10": round(recall[10] / q_n, 3), "mrr": round(mrr / q_n, 3),
        "chunk_hit@1": round(chunk_hit[1] / q_n, 3), "chunk_hit@5": round(chunk_hit[5] / q_n, 3),
    }
    depth_rows = [
        {"model": cfg["model"], "dims": dims_label, "chunk_size": chunk_size, "depth": dd,
         "recall@5": round(v["r5"] / v["n"], 3), "chunk_hit@5": round(v["c5"] / v["n"], 3), "n": v["n"]}
        for dd, v in sorted(by_depth.items())
    ]
    return row, depth_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true",
                    help="merge into existing results (keep rows for models not run this pass)")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    docs, queries = load_corpus()
    print(f"corpus: {len(docs)} docs, {len(queries)} queries")

    rows, depth_rows, skipped = [], [], []
    for cfg in CONFIGS:
        try:
            served = embed_client.models(cfg["endpoint"])
        except Exception as e:  # noqa: BLE001
            skipped.append({"model": cfg["model"], "reason": f"endpoint down: {e!r}"})
            print(f"SKIP {cfg['model']}: endpoint {cfg['endpoint']} not reachable ({e!r})")
            continue
        print(f"{cfg['model']} up at {cfg['endpoint']} (serving {served})")
        for dims in cfg["dims"]:
            for cs in cfg["chunk_sizes"]:
                tag = f"{cfg['model']} dims={dims} chunk={cs}"
                try:
                    row, drows = run_one(cfg, dims, cs, docs, queries)
                except Exception as e:  # noqa: BLE001
                    skipped.append({"model": cfg["model"], "dims": dims, "chunk_size": cs, "reason": repr(e)})
                    print(f"  FAIL {tag}: {e!r}")
                    continue
                rows.append(row)
                depth_rows.extend(drows)
                print(f"  {tag}: dim={row['vector_dim']} chunks/doc={row['chunks_per_doc']} "
                      f"tok/s={row['tokens_per_sec']} R@1={row['recall@1']} R@5={row['recall@5']} "
                      f"chunkhit@5={row['chunk_hit@5']}")

    if args.merge and (RESULTS / "summary.json").exists():
        prev = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
        ck = lambda r: (r["model"], str(r["dims"]), r["chunk_size"])  # noqa: E731
        dk = lambda r: (r["model"], str(r["dims"]), r["chunk_size"], r["depth"])  # noqa: E731
        new_ck = {ck(r) for r in rows}
        new_dk = {dk(r) for r in depth_rows}
        rows = [r for r in prev.get("configs", []) if ck(r) not in new_ck] + rows
        depth_rows = [r for r in prev.get("by_depth", []) if dk(r) not in new_dk] + depth_rows
    rows.sort(key=lambda r: (r["model"], str(r["dims"]), r["chunk_size"]))
    depth_rows.sort(key=lambda r: (r["model"], str(r["dims"]), r["chunk_size"], r["depth"]))

    _write_csv(RESULTS / "by_config.csv", rows)
    _write_csv(RESULTS / "by_depth.csv", depth_rows)
    (RESULTS / "summary.json").write_text(
        json.dumps({"configs": rows, "by_depth": depth_rows, "skipped": skipped,
                    "n_docs": len(docs), "n_queries": len(queries)}, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {len(rows)} config rows -> {RESULTS}")
    if skipped:
        print(f"skipped: {skipped}")


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
