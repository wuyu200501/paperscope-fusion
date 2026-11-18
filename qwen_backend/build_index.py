"""Embed QASPER chunks via local Ollama and store in ChromaDB (persistent).

One Chroma collection per chunking strategy:
  qasper_A  <- chunks_A.jsonl (structured)
  qasper_B  <- chunks_B.jsonl (sliding window)

Notes
  - documents are embedded WITHOUT any instruction prefix; the task
    instruction is only prepended to *queries* (see embed_query in
    retrieve.py), per the Qwen3-Embedding model card.
  - idempotent: a collection is rebuilt only when its chunk count differs.

Usage (desktop):
  python build_index.py --chunks F:/AI/cs5481-project/chunks --db F:/AI/cs5481-project/chroma
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path


def embed_ollama(texts: list[str], model: str, host: str) -> list[list[float]]:
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/embed", data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())
    if data.get("error"):
        raise RuntimeError(f"ollama embed error: {data['error']}")
    embs = data["embeddings"]
    if len(embs) != len(texts):
        raise RuntimeError(f"expected {len(texts)} embeddings, got {len(embs)}")
    return embs


def build_collection(client, name: str, rows: list[dict], model: str, host: str, batch: int):
    existing = {c.name: c.count() for c in client.list_collections()}
    if existing.get(name) == len(rows):
        print(f"[{name}] already indexed ({len(rows)} chunks), skip")
        return
    if name in existing:
        client.delete_collection(name)
    col = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

    t0 = time.time()
    for i in range(0, len(rows), batch):
        part = rows[i:i + batch]
        embs = embed_ollama([r["text"] for r in part], model, host)
        col.add(
            ids=[r["chunk_id"] for r in part],
            embeddings=embs,
            documents=[r["text"] for r in part],
            metadatas=[
                {
                    "paper_id": r["paper_id"],
                    "section_name": r["section_name"],
                    "n_tokens": r.get("n_tokens", 0),
                }
                for r in part
            ],
        )
        done = min(i + batch, len(rows))
        print(f"[{name}] {done}/{len(rows)} chunks ({done / (time.time() - t0):.0f}/s)", flush=True)
    print(f"[{name}] done in {time.time() - t0:.0f}s")


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=Path, default=Path("chunks"))
    ap.add_argument("--db", type=Path, default=Path("chroma"))
    ap.add_argument("--model", default="qwen3-embedding:4b-fp16")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    import chromadb

    client = chromadb.PersistentClient(path=str(args.db))
    for strategy in ("A", "B"):
        path = args.chunks / f"chunks_{strategy}.jsonl"
        if not path.is_file() or not path.stat().st_size:
            print(f"missing or empty {path}, skipping strategy {strategy}", file=sys.stderr)
            continue
        rows = load_rows(path)
        build_collection(client, f"qasper_{strategy}", rows, args.model, args.host, args.batch)


if __name__ == "__main__":
    main()
