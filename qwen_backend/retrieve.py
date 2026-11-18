"""Semantic retrieval against the QASPER Chroma collections.

Queries are embedded with the Qwen3-Embedding task instruction prefix
(documents were embedded without it), per the model card.

CLI:
  python retrieve.py "what optimizer does BERT use?" --strategy A --k 5
"""
from __future__ import annotations

import argparse
import json
import urllib.request

QUERY_INSTRUCTION = (
    "Given a user question about a scientific paper, retrieve the passages "
    "from that paper that help answer the question."
)
DEFAULT_MODEL = "qwen3-embedding:4b-fp16"
DEFAULT_HOST = "http://localhost:11434"


def _embed(texts: list[str], model: str, host: str) -> list[list[float]]:
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/embed", data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    if data.get("error"):
        raise RuntimeError(f"ollama embed error: {data['error']}")
    return data["embeddings"]


def embed_query(question: str, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST) -> list[float]:
    """Query embedding WITH the mandatory instruction prefix (1-5% recall otherwise)."""
    return _embed([f"Instruct: {QUERY_INSTRUCTION}\nQuery: {question}"], model, host)[0]


def retrieve(collection, question: str, k: int = 5, model: str = DEFAULT_MODEL,
             host: str = DEFAULT_HOST, paper_id: str | None = None,
             rerank_n: int = 0) -> list[dict]:
    """Top-k chunks; each result gets `score` used for ordering.

    paper_id restricts retrieval to one paper (metadata filter) — the
    per-paper "user is reading this paper" setting; None = global search.
    rerank_n > 0: fetch that many candidates and re-order them with the
    Qwen3-Reranker cross-encoder before returning the top-k; `score`
    becomes the reranker probability (bi-encoder score kept as `bi_score`).
    """
    emb = embed_query(question, model, host)
    where = {"paper_id": paper_id} if paper_id else None
    n = max(k, rerank_n)
    res = collection.query(query_embeddings=[emb], n_results=n, where=where)
    out = []
    for i, cid in enumerate(res["ids"][0]):
        out.append({
            "chunk_id": cid,
            "text": res["documents"][0][i],
            "meta": res["metadatas"][0][i],
            "score": 1.0 - res["distances"][0][i],
        })
    if rerank_n:
        from .rerank import rerank
        for r, s in zip(out, rerank(question, [r["text"] for r in out])):
            r["bi_score"] = r["score"]
            r["score"] = s
        out.sort(key=lambda r: -r["score"])
    return out[:k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--strategy", default="A", choices=["A", "B"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--db", default="data/chroma")
    args = ap.parse_args()

    import chromadb

    col = chromadb.PersistentClient(path=args.db).get_collection(f"qasper_{args.strategy}")
    hits = retrieve(col, args.question, k=args.k)
    for h in hits:
        print(f"[{h['score']:.3f}] {h['chunk_id']} | {h['meta']['paper_id']} | {h['meta']['section_name']}")
        print(f"  {h['text'][:160]}...")


if __name__ == "__main__":
    main()
